import xml.etree.ElementTree as ET
from re import compile as rc
from ..cluster import cluster
from hashlib import md5
import numpy as np
import os
from subprocess import call, PIPE
from DyMat import DyMatFile
import pandas as pd
from .py_api import Model_api

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

class open_modelica(Model_api):
    def __init__(self, files, model,
                 force_recompilation=False,
                 abort_slow=0,
                 solver='dassl',
                 tmp_storage=None,
                 init_statei=None):

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
        self.viz_dir = prefix + f"lofi_viz_data"
        self.viz_file = self.viz_dir + f"/{self.model}.h5"

        # JIT compile if necessary (or forced)
        self.force_recompilation = force_recompilation
        if self.model_changed() or self.force_recompilation:
            self.compile()
        self.make_dir(self.result_dir)
        self.make_dir(self.viz_dir, clear=False)

        self.abort_slow = abort_slow
        self.solver = solver
        cluster.comm.Barrier()  # synchronize here

        # read init_file for information about the compiled model
        self.xml_info = xml_init_file_handler(self.init_file)

        # get parameter info from within parsed init_file
        # this needs to be here before call to super().__init__()
        self.y_names = self.xml_info.get_objectives("name")
        self.p_names = self.xml_info.get_parameters("name")
        self.p_lb = np.array(self.xml_info.get_parameters("lower_bound"))
        self.p_ub = np.array(self.xml_info.get_parameters("upper_bound"))
        self.p_start = self.xml_info.get_parameters("start")

        # get variables to be plotted from withing init_file
        self.plot_vars = self.xml_info.get_plot_variables("name")

        super().__init__()

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
    def make_dir(self, dir, clear=True):
        """Creates folder of given name"""
        if not os.path.exists(dir):
            os.mkdir(dir)
        elif clear:
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

    # simulation related functions
    def get_simulation_command_root(self):
        """Construct the basic flags/options for the model executable"""
        flags = f" -inputPath={self.compile_dir}"
        flags += " -lv=-LOG_STATS,-stdout,-assert"
        flags += f" -s={self.solver}"
        flags += " -cpu"
        if self.abort_slow > 0:
            flags += f" -alarm={self.abort_slow}"
        return self.compiled_file + flags

    def get_result_path(self, id, key="tag"):
        """Construct result path based on situation"""
        return self.result_dir + f"/{key}_{id}.mat"

    def result_file_flag(self, id, key="tag"):
        """Constructs result path flag for the model executable"""
        return " -r=" + self.get_result_path(id)

    def override_parameters(self, p):
        """Construcst flag that overrides parameters in the model executable"""
        flag = " -override "
        for i in range(self.m):
            flag += f"{self.p_names[i]}={p[i]},"
        return flag

    def evaluate(self, prms, result_tag):
        """Build shell command with all flags and runs the executable"""
        command = self.get_simulation_command_root()
        command += self.result_file_flag(result_tag)
        command += self.override_parameters(prms)
        return call(command, shell=True, stdout=PIPE, stderr=PIPE)

    def get_tagged_result(self, id, key="tag"):
        """Loads the result file dropped by the executable"""
        return DyMatFile(self.get_result_path(id))

    def extract_raw_loss(self, result):
        """Extracts arrays coresponding to all vars. marked as #objective"""
        return result.getVarArray(self.y_names, withAbscissa=False)[0:,:]

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













