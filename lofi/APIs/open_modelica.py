import xml.etree.ElementTree as ET
from re import compile as rc
from ..cluster import cluster
from hashlib import md5
import numpy as np
import os
from subprocess import call, PIPE
from DyMat import DyMatFile
import pandas as pd
from ..visualizers.visualizers import default_visual_callback

class xml_init_file_handler():
    def __init__(self, init_file):
        self.init_file = init_file
        self.tree = ET.parse(self.init_file)
        self.all = self.parse_xml()

    def parse_xml(self):
        all = []
        if os.path.exists(self.init_file):
            self.root = self.tree.getroot()
            hashtag = rc(r"#(\w+)")

            for sv in self.root.iter("ScalarVariable"):
                dict = {}
                dict["name"] = sv.get("name")
                dict["description"] = sv.get("description")
                if dict["description"] is not None:
                    dict["hashtags"] = hashtag.findall(dict["description"])
                else:
                    dict["hashtags"] = []
                dict["variability"] = sv.get("variability")
                dict["classType"] = sv.get("classType")

                ch = sv.getchildren()
                start = None
                lb = None
                up = None
                for att in ch:
                    start = att.get('start')
                    lb = att.get("min")
                    ub = att.get("max")
                try:
                    dict["start"] = float(start)
                except:
                    dict["start"] = None
                dict["lower_bound"] = float(lb) if lb is not None else -1.0
                dict["upper_bound"] = float(ub) if ub is not None else 1.0
                all.append(dict)
            return all

    def get_parameters(self, info_key):
        parameters = []
        for sv in self.all:
            passes = sv["variability"] == "parameter"
            passes = passes and sv["start"] is not None
            passes = passes and sv["classType"] != "rSen"
            passes = passes and any(tag in sv["hashtags"] for tag in ["optimize","optimise"])
            if passes:
                parameters.append(sv[info_key])
        return parameters

    def get_objectives(self, info_key):
        objectives = []
        for sv in self.all:
            passes = sv["variability"] == "continuous"
            passes = passes and "der(" not in sv["name"]
            passes = passes and any(tag in sv["hashtags"] for tag in ["objective"])
            if passes:
                objectives.append(sv[info_key])
        return objectives

    def get_plot_variables(self, info_key):
        variables = []
        for sv in self.all:
            passes = sv["variability"] == "continuous"
            passes = passes and "der(" not in sv["name"]
            passes = passes and any(tag in sv["hashtags"] for tag in ["plot"])
            if passes:
                variables.append(sv[info_key])
        return variables

    def write_new_setup(self, output_path):
        if cluster.global_rank == 0:
            self.tree.write(output_path)


class open_modelica():
    def __init__(self, files, model,
                 force_recompilation=False,
                 abort_slow=0,
                 solver='dassl',
                 tmp_storage=None):

        # resolve file paths
        files = [files] if type(files) is not list else files
        self.files = [os.path.abspath(file) for file in files]
        self.model = model
        if tmp_storage is not None:
            prefix = tmp_storage
        if os.path.isdir('/dev/shm'):
            prefix = '/dev/shm/'
        else:
            prefix = os.getcwd()
        self.compile_dir = prefix + f"/compiled_" + self.model
        self.result_dir = prefix + f"/result_" + self.model
        self.compiled_file = self.compile_dir + f"/{self.model}"
        self.init_file = self.compiled_file + "_init.xml"
        self.viz_file = prefix + f"lofi_viz_data/{self.model}.h5"

        # JIT compile if necessary (or forced)
        self.force_recompilation = force_recompilation
        if self.model_changed() or self.force_recompilation:
            self.compile()
        self.make_dir(self.result_dir)

        self.abort_slow = abort_slow
        self.solver = solver
        cluster.comm.Barrier()  # synchronize here

        # read init_file for information about the compiled model
        self.xml_info = xml_init_file_handler(self.init_file)

        # get objectives from within parsed init_file
        self.y_names = self.xml_info.get_objectives("name")
        self.y_len = len(self.y_names)

        # get parameters from within parsed init_file
        self.p_names = self.xml_info.get_parameters("name")
        self.m = len(self.p_names)
        self.p_lb = np.array(self.xml_info.get_parameters("lower_bound"))
        self.p_ub = np.array(self.xml_info.get_parameters("upper_bound"))
        self.p_start = self.xml_info.get_parameters("start")

        # get variables to be plotted from withing init_file
        self.plot_vars = self.xml_info.get_plot_variables("name")

        # evaluation counter
        self.evals = 0

        # model state with res_file reference info
        if cluster.global_rank == 0:
            self.p = self.p_start
            self.y = self.loss(self.p, result_tag_override=0)[0]
            self.result = self.read_result_file(0)
        self.result_owner = None
        self.result_id = None
        self.new_best_result = False

        self.log = pd.DataFrame(columns=['evals',
                                         'loss'])

    def hash(self):
        """Calculates md5 hash of the model .mo files"""
        hash_md5 = md5()
        for file in self.files:
            with open(file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @cluster.on_master
    def write_hash(self):
        """Writes the hash of the .mo file to disk"""
        with open(self.compiled_file + "_hash.txt", "w") as f:
            f.write(self.hash())

    def read_hash(self):
        """Reads the hash of the .mo file from disk if it exists"""
        if os.path.exists(self.compiled_file + "_hash.txt"):
            with open(self.compiled_file + "_hash.txt", "r") as f:
                return f.read()

    def model_changed(self):
        """Compares old and new hash of the .mo files. Returns True if they changed"""
        return not os.path.exists(self.compile_dir) or not self.read_hash() == self.hash()

    @cluster.on_master
    def make_dir(self, dir):
        """Creates folder of given name"""
        if not os.path.exists(dir):
            os.mkdir(dir)
        else:
            os.system("rm " + dir + "/*")

    @cluster.on_machine
    def compile(self):
        """Compiles the model on each machine with OMC that they have"""
        self.make_dir(self.compile_dir)
        script_content = f"cd(\"{self.compile_dir}\");\n"
        script_content += "getErrorString();\n"
        script_content += "loadModel(Modelica);\n"
        script_content += "getErrorString();\n"
        for file in self.files:
            script_content += f"loadFile(\"{file}\");\n"
            script_content += "getErrorString();\n"
        script_content += f"buildModel({self.model});\n"
        script_content += "getErrorString();\n"
        script_content += "//comment\n"
        script_file = self.compiled_file + ".mos"
        with open(script_file, "w") as OMC_script:
            OMC_script.write(script_content)
        print(f"Compiling OpenModelica model... ")
        command = "omc " + script_file
        os.system(command)
        self.write_hash()

    def get_simulation_command_root(self):
        """Construct the basic flags/options for the model executable"""
        flags = f" -inputPath={self.compile_dir}"
        flags += " -lv=-LOG_STATS,-stdout,-assert"
        flags += f" -s={self.solver}"
        if self.abort_slow > 0:
            flags += f" -alarm={self.abort_slow}"
        return self.compiled_file + flags

    def get_result_path(self, id, key="tag"):
        """Construct result path based on situation"""
        return self.result_dir + f"/{key}_{id}.mat"

    def result_file_flag(self, id, key="tag"):
        """Constructs result path for the model executable"""
        return " -r=" + self.get_result_path(id)

    def read_result_file(self, id, key="tag"):
        return DyMatFile(self.get_result_path(id))

    def override_parameters(self, p):
        """Construcst flag that overrides parameters in the model executable"""
        flag = " -override "
        for i in range(self.m):
            flag += f"{self.p_names[i]}={p[i]},"
        return flag

    @cluster.on_master
    def get_formatted_parameters(self):
        """Returns string of best known parameters in OM override_file format"""

        formated_parameters = ""
        for name, value in zip(self.p_names, self.p):
            formated_parameters += f"{name}={value}\n"
        return formated_parameters

    @cluster.on_master
    def save_parameters(self):
        """Saves the currently best known parameters in OM override_file format"""

        with open(self.model + "_solution.txt", "w") as f:
            f.write(self.get_formatted_parameters())

    @cluster.on_master
    def print_parameters(self):
        """Prints the best known parameters in OM override_file format"""

        print(self.get_formatted_parameters())

    def call_simulation(self, prms, result_tag):
        command = self.get_simulation_command_root()
        command += self.result_file_flag(result_tag)
        command += self.override_parameters(prms)
        return call(command, shell=True, stdout=PIPE, stderr=PIPE)

    def extract_raw_loss(self, result):
        return result.getVarArray(self.y_names, withAbscissa=False)[0:,:]

    @cluster.on_master
    def get_formated_loss(self, f=np.max):
        "Returns string of the best known losses"
        formated_loss = ""
        y = self.extract_raw_loss(self.get_result()) 
        for name, value in zip(self.y_names, f(y, axis=0)):
            formated_loss += f"{name}={value}\n"
        return formated_loss

    def print_loss(self, f=np.max):
        if cluster.global_rank == 0:
            print(self.get_formated_loss(f=f))

    def save_loss(self, f=np.max):
        # master opens file and writes the content into it
        if cluster.global_rank == 0:
            with open(self.model + "_loss.txt", "w") as f:
                f.write(self.get_formatted_loss(f=f))

    def inf_loss(self, prms, timer):
        return np.inf, np.sum(np.abs(prms)), 1, timer.get_elapsed(), cluster.global_rank

    def real_loss(self, y, prms, timer):
        return np.sum(y), np.sum(np.abs(prms)), 0, timer.get_elapsed(), cluster.global_rank

    def loss(self, prms, result_tag_override=None):
        """Executes model with given parameters and returns tuple of sim. data"""

        timer = cluster.timer()
        self.evals += 1

        # resolve name (tag/id) of the result file
        if result_tag_override is not None:
            result_tag = result_tag_override
        else:
            result_tag = cluster.status.Get_tag()

        retcode = self.call_simulation(prms, result_tag)

        if retcode != 0:
            return self.inf_loss(prms, timer)
        else:
            result = self.read_result_file(result_tag)
            y = self.extract_raw_loss(result)  # y is a np.array of floats

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return self.inf_loss(prms, timer)
        else:
            return self.real_loss(y, prms, timer)

    @cluster.on_master
    def update_state(self, p, y, owner=None, res_id=None):
        "This function updates the current best state of the model"
        self.y = y
        self.p[:] = p
        self.save_parameters()
        self.result_owner = owner
        self.result_id = res_id 
        self.new_best_result = True

    @cluster.on_master
    def update_viz_file(self, res):
    
        store = pd.HDFStore(self.viz_file)

        #model data
        t = res.getVarArray([self.plot_vars[0]])[0,:]
        model_df = pd.DataFrame(index=t)
        for name in self.plot_vars:
            model_df[name] = res.getVarArray([name])[1,:]
        store['model_df'] = model_df

        #log data
        self.log.loc[len(self.log)] = [self.total_evals, self.y]
        store['api_df'] = self.log
        store.close()

    def update_log(self):
        self.total_evals = self.get_total_evals()
        res = self.get_result()
        try:
            self.update_viz_file(res)
        except:
            None

    def get_total_evals(self):
        """Returns the number of calls to loss function"""
        return cluster.sum_all(self.evals)

    def pull_result(self):
        """Updates best known result data on master node by pulling from cluster"""
        
        # Tell every node whether new result is available
        self.new_best_result = cluster.broadcast(self.new_best_result)

        # If newer resuls is available, pull it to master node
        if self.new_best_result:

            # Tell every node what is the best result_id and who owns that file
            data = (self.result_id, self.result_owner)
            self.result_id, self.result_owner = cluster.broadcast(data)

            # If a node is the owner of the wanted file, it reads the data and
            # sends the data to the master node (the zero node)
            if self.result_owner == cluster.global_rank:
                result = self.read_result_file(self.result_id)
                cluster.comm.send(result, 0)

            # Master node receives data from the owner of new best results
            if cluster.global_rank == 0:
                self.result = cluster.comm.recv(None, source=self.result_owner)

            # Every node gets notified that newest results have been
            # successfully delivered to the master node
            self.new_best_result = False

    def get_result(self):
        """Returns best of results found so far"""

        # make sure that master has the newest best result
        self.pull_result()
        
        if cluster.global_rank == 0:
            return self.result


