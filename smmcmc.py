import subprocess
import numpy as np
import copy
import emcee
import os
import pty
import re
import csv
from tqdm import tqdm


class smmcmc:
    def __init__(self, SphenoBlocksDict, ExpectedDataDict, ConstraintsBeforeSPheno, ConstraintsAfterSPheno,
                SPhenoFilePath, SPhenoInputFilePath, UseMicrOmegas = False, MicrOmegasFilePath = " ",
                nWalkers = 100, LikelihoodThreshold = 0.9, AcceptedPoints = 1000, OutputMCMCfile = "output_mcmc.csv",
                Steps = None, Stretch = 1.2, LogParameterization = False, StrictParametersRanges = True, WriteOnlyAccepted = True):

        """
        The `smmcmc` class allows for the exploration of the parameter space of a scotogenic model using the Markov Chain Monte Carlo (MCMC) method
        and the computational tools SPheno and, optionally, MicrOmegas. SPheno is essential for performing calculations related to masses,
        branching ratios, and other relevant parameters, while MicrOmegas is employed, if enabled, to study dark matter properties such as relic density.

        For the correct functioning of this class, it is essential that the physical model is fully implemented and functional in both SPheno
        and, if applicable, MicrOmegas. This includes ensuring that input configurations and definitions are properly set up in both softwares.

        ### Arguments:

        ### SphenoBlocksDict (dict):
        This argument is a dictionary that defines the model parameters to be explored during the analysis.
        It is a nested dictionary structured into blocks that correspond to the input blocks in the SPheno input file.
        Each block is represented by an internal dictionary, where:

        - The *keys* correspond to the names of the blocks as defined in the SPheno input file (e.g., BLOCK MINPAR, "MINPAR").
        - The *values* within each block are the parameters to be explored. These parameters can be defined in three different ways:

            1. **List:** If the parameter should vary within an interval, a list with two elements is provided:
            the minimum and maximum values of the interval. If you wish to strictly restrict the generated values to the specified range,
            the `StrictParametersRanges` argument can be enabled.

            2. **Float:** If the parameter should remain constant during the analysis, it is defined directly
            with its numerical value.

            3. **Function:** If the value of a parameter must be calculated dynamically based on other parameters, it is defined as a function.
            This function must be defined beforehand and must take as an argument a dictionary containing all model parameters, accessible via
            their block names. The function must return a single float value. For example:

            ```python
            def lambda1_fixed(parameters: dict):
                    other = parameters["OTHER"]
                    lambda_1 = (other["mH1"]**2) * (np.cos(other["alpha"])**2)
                    lambda_1 += (other["mH2"]**2) * (np.sin(other["alpha"])**2)
                    lambda_1 = lambda_1/(other["v"]**2)
                    return lambda_1
            ```

            If you wish to calculate multiple parameters at once with a single function, at least one parameter inside a block must be defined
            as a function. You cannot define the name of a block directly as a function, as this would generate an error. Instead, a
            parameter within the block should be named as a function, and this must return a dictionary containing the blocks and parameters to be calculated.
            As with the single parameter case, this function must be defined beforehand and must take as an argument a dictionary containing all model parameters,
            accessible via their block names. For example:

                ```python
                def calculate_Yn(parameters: dict):
                    return {
                        "COUPLINGSYNIN": {
                            "Yn(1,1)": 1.0,
                            "Yn(1,2)": 1.0,
                            "Yn(2,1)": 1.0,
                            "Yn(2,2)": 1.0,
                        },
                        "IMCOUPLINGSYNIN": {
                            "IYn(1,1)": 1.0,
                            "IYn(1,2)": 1.0,
                            "IYn(2,1)": 1.0,
                            "IYn(2,2)": 1.0,
                        }
                    }
            ```

        Additionally, you may include extra parameters not found in the SPheno blocks but necessary for calculations.
        These can be grouped in custom blocks with arbitrary names (e.g., "OTHER") to organize fixed or generated values
        used in functions.

        #### Example of `SphenoBlocksDict`:
        Below is a properly structured example dictionary for this argument:

        ```python
        SphenoBlocksDict = {
            "MINPAR": {
                "lambda1Input": lambda1_fixed,
                "lam4EtaInput": [1e-6, 1],
                "lamHEta3Input": [1e-6, 1],
                "lambda4HEtaInput": [1e-6, 1],
                "lambda5HEtaInput": [1e-8, 1],
                "lam4SigInput": lambda_sigma_fixed,
                "lamHSigInput": lambda3_hsigma_fixed,
                "lamEtaSigInput": [1e-6, 1],
                "vSiInput": [0.5e3, 10e3],
                "mEta2Input": [1e5, 1e7]
            },
            "OTHER": {
                "alpha": [0.0, np.pi / 2],
                "mH2": 500.0,
                "mH1": 125.0,
                "v": 246.0
            }
        }
        ```

        This example includes two main blocks: "MINPAR", which contains parameters directly used in SPheno, and "OTHER",
        which defines additional parameters needed for specific calculations (such as constants or input values for dependent functions).

        #### Additional considerations:

        - Parameter names must exactly match those defined in the SPheno input files.
        - Functions used to calculate dynamic parameters must be properly documented and validated, as errors
        in their definition can produce unexpected results during the analysis.

        ### ExpectedDataDict (dict):

        This argument is a dictionary that defines the quantities used to evaluate the likelihood of the parameters in the analysis. It is used to read specific data
        from the output files generated by SPheno and, optionally, by MicrOmegas, in order to compute the likelihood of the parameters in each iteration
        of the MCMC process. It also allows for extracting other relevant quantities for additional analysis or applying constraints.

        The structure of this dictionary is similar to that of 'SphenoBlocksDict', as it is organized into blocks. Each block is represented by an internal dictionary structured as follows:

        1. **Keys of the main dictionary:**
            The keys correspond to the block names in the output files of SPheno or MicrOmegas. For example:
            - In the case of SPheno, a block could be "MASS" (equivalent to BLOCK MASS in the output file).
            - For decays, use "DECAY No", where `No` is the numerical identifier of the decay.
            - If MicrOmegas is included, a block named "MICROMEGAS" must be specified.

        2. **Internal dictionaries (associated values for each block):**
            Each block contains a dictionary with the following elements:
            - The *keys* represent the quantities of interest.
            - The *values* can take two formats depending on the purpose:
                - **List:** If the quantity will be used for likelihood calculation, the value must be a list with two elements:
                    1. The expected value (from literature or experiments).
                    2. Its associated uncertainty.
                - **String (str):** If the quantity is not directly used in likelihood calculation but is needed for additional analysis, the value should be a string.
                It is recommended that the string represent a number (e.g., "0.0") to ensure that a default value is assigned if the quantity is not found in the output file.
                - **Function:** Functions can be defined to dynamically compute the likelihood values. This allows evaluating likelihood at each iteration
                of the MCMC process without relying on fixed values. The function must be defined beforehand and must take as an argument a dictionary containing all
                the parameters included in this dictionary (read from SPheno or MicrOmegas), accessible by their block names.
                    1. To compute a single quantity, the function must return a list with two elements [expected value, uncertainty].
                    2. To compute multiple quantities at once, the function must return a dictionary where keys are the quantity names and values are:
                        a. A list [expected value, uncertainty] if the quantity contributes to likelihood.
                        b. A float if the quantity is only to be extracted for additional analysis.

                    For example:
                    ```python
                    def dynamic_quantities(parameters: dict):
                        return {
                            "Mass_H1": [parameters["mH1"], 0.05 * parameters["mH1"]],
                            "Mass_H2": [parameters["mH2"], 0.05 * parameters["mH2"]],
                            "Mixing_Angle": parameters["alpha"]
                        }
                    ```

        #### Additional details:
        - All values specified in this dictionary that are present in the outputs from SPheno or MicrOmegas will be included in the output file generated by the analysis. This is useful for storing
        additional information that is not used in the likelihood calculation but may be relevant for other analyses.
        - For decays, blocks must include the prefix "DECAY " followed by the identifier of the particle. For example:
        ```python
        "DECAY 25": {
            "WIDTH": [3.7e-3, 1.7e-3],
            "BR(hh_1 -> Ah_2 Ah_2)": "0.0",
            "BR(hh_1 -> FX0_1 FX0_1)": "0.0",
        }
        ```

        If using MicrOmegas, a block named "MICROMEGAS" must be included with the specific quantities to extract. For example:

        ```python
        "MICROMEGAS": {
            "Xf (Freeze-out temperature)": "0.0",
            "Omega h^2 (Dark matter relic density)": [0.120, 0.0036]
        }
        ```

        ### ConstraintsBeforeSPheno (list):

        This argument allows defining constraints that must be verified before running SPheno at each iteration of the analysis. These constraints are applied to the parameters
        in `SphenoBlocksDict`. Implementing constraints in this way results in a more efficient analysis since it avoids running SPheno and/or MicrOmegas for
        iterations where input parameters already fail to meet defined restrictions.

        #### Definition:
        `ConstraintsBeforeSPheno` is a list of previously defined functions. Each function must meet the following characteristics:
        1. **Receive a single argument:** 
        A dictionary following the same structure as `SphenoBlocksDict`. It cannot include elements not previously named there.
        2. **Return a boolean value:**
        - Returns `True` if the parameters meet the evaluated restriction.
        - Returns `False` if the restriction is not met.

        #### Example:
        Detailed example of defining and using `ConstraintsBeforeSPheno`:

        ```python
        def chi_masses(blocks):
            kappain = blocks["KAPPAIN"]
            chi_1 = kappain["Kap(1,1)"]
            chi_2 = kappain["Kap(2,2)"]
            chi_3 = kappain["Kap(3,3)"]
            return (chi_1 < chi_2 and chi_2 < chi_3)
        ```

        ```python
        ConstraintsBeforeSPheno = [chi_masses]
        ```

        ### ConstraintsAfterSPheno (list):

        This argument allows defining constraints that must be verified after running SPheno at each iteration of the analysis. These constraints are applied to the parameters
        in `ExpectedDataDict` and/or `SphenoBlocksDict`.

        #### Definition:
        `ConstraintsAfterSPheno` is a list of previously defined functions. Each function must meet the following characteristics:
        1. **Receive a single argument:** 
        A dictionary following the structure of `ExpectedDataDict` and/or `SphenoBlocksDict`. It cannot include elements not previously named there.
        2. **Return a boolean value:**
        - Returns `True` if the parameters meet the evaluated restriction.
        - Returns `False` if the restriction is not met.

        #### Example:
        Detailed example of defining and using `ConstraintsAfterSPheno`:

        ```python
        def branching_ratios(model_decay: dict):
            alpha = model_decay["OTHER"]["alpha"]
            model_decay_ = model_decay["DECAY 25"]
            br1 = float(model_decay_["BR(hh_1 -> Ah_2 Ah_2 )"])
            br2 = float(model_decay_["BR(hh_1 -> FX0_1 FX0_1 )"])
            br3 = float(model_decay_["BR(hh_2 -> FX0_2 FX0_2 )"])
            br4 = float(model_decay_["BR(hh_2 -> FX0_3 FX0_3 )"])
            return ((np.cos(alpha)**2) * (br1 + br2 + br3 + br4)) < 0.19
        ```

        ```python
        ConstraintsAfterSPheno = [branching_ratios]
        ```

        ### SPhenoFilePath (string):

        This parameter indicates the absolute path to the SPheno model executable file.

        Example:
        ```python
        SPhenoFilePath = "/home/f.herreras/SPheno-4.0.5/bin/SPhenoScotogenic_SLNV"
        ```

        ### SPhenoInputFilePath (string):

        This parameter indicates the absolute path to the SPheno model input file.

        Example:
        ```python
        SPhenoInputFilePath = "/home/f.herreras/SPheno-4.0.5/Scoto_SLNV_11.10/Input_Files/LesHouches.in.Scotogenic_SLNV"
        ```

        ### UseMicrOmegas (boolean):

        This parameter indicates whether to use MicrOmegas in the analysis. By default, it is False, but if set to True, the code will call MicrOmegas in each cycle using
        the output from SPheno for that cycle.

        ### MicrOmegasFilePath (string):

        If MicrOmegas is used, this parameter must specify the absolute path to the executable of the model in MicrOmegas.

        Example:
        ```python
        MicrOmegasFilePath = "/home/f.herreras/micromegas_6.0.5/Scoto_SLNV_11.10/CalcOmega_MOv5"
        ```

        ### nWalkers (int):

        Indicates the number of walkers in the ensemble sampling of the Markov Chain Monte Carlo.

        ### LikelihoodThreshold (float):

        Indicates the minimum likelihood that parameters must have to count as an accepted point (not to be confused with the acceptance probability of a parameter set).

        ### AcceptedPoints (int):

        Indicates the number of accepted points required to complete the analysis.

        ### OutputMCMCfile (string):

        Indicates the name of the file containing the points returned by the analysis.

        ### Steps (False or int):

        If a maximum number of cycles is also desired in addition to a target number of accepted points, the number should be specified. Otherwise, leave as `False`.

        ### Stretch (float):

        In the ensemble sampling method used by emcee, the stretch factor is a value that controls the size of proposed steps in parameter space.

        In the Stretch Move, each walker proposes a new point based on the position of other walkers. The new point is generated as:
            y = x + Z(x' - x)

        Where:
        - x is the current position of the walker.
        - x' is a randomly selected position from another walker.
        - Z is a random scaling factor depending on the stretch factor a, defined as:
            Z = Uniform[1/a, a]
        This Z determines how much the step is "stretched" or "contracted" towards the other walker, hence the name Stretch Move.

        - Small stretch values: Produce small steps (less exploration, more local).
        - Large stretch values: Produce large steps (greater exploration, but potentially inefficient in complex parameter spaces).

        ### LogParameterization (boolean):

        When using emcee, it is recommended to work with numbers in the [0, 1] interval, which implies rescaling the data. This rescaling
        can be of two types: uniform (if `False` is specified) or logarithmic (if `True` is specified). The choice between these methods depends
        on the specific advantages each offers depending on the context of the problem. Both approaches are valid.

        ### StrictParametersRanges (boolean):

        Indicates whether the parameters in `SphenoBlocksDict` defined as intervals must strictly respect their limits.  
        - If set to `False`, parameters may go beyond the defined interval limits during exploration.  
        - If set to `True`, parameters will remain strictly within the defined bounds.

        ### WriteOnlyAccepted (boolean):

        Determines which data are saved in the output file:  
        - If set to `True`, only accepted data with likelihood above `LikelihoodThreshold` are saved.  
        - If set to `False`, all explored MCMC data are saved, regardless of likelihood value.
        """
        self.spheno_block_dict = SphenoBlocksDict  
        self.expected_data_dict = ExpectedDataDict
        self.constraints_before_spheno = ConstraintsBeforeSPheno 
        self.constraints_after_spheno = ConstraintsAfterSPheno 
        self.spheno_file_path = SPhenoFilePath 
        self.spheno_input_file_path = SPhenoInputFilePath 
        self.use_micromegas = UseMicrOmegas 
        self.micromegas_file_path = MicrOmegasFilePath 
        self.nwalkers = nWalkers 
        self.likelihood_threshold = np.log(LikelihoodThreshold)    
        self.accepted_points = AcceptedPoints 
        self.output_mcmc_file = OutputMCMCfile 
        self.steps = Steps
        self.stretch = Stretch      
        self.log_parameterization = LogParameterization 
        self.strict_parameters_ranges = StrictParametersRanges 
        self.write_only_accepted = WriteOnlyAccepted
        self.dinamic_likelihood = False
        self.spheno_output_file_path = self.spheno_input_file_path.replace("LesHouches.in", "SPheno.spc")


    
    @staticmethod
    def dict_to_vector(blocks_dict):


        main_parameters_values = []
        main_parameters_names = []
        extra_parameters_values = []
        extra_parameters_names = []
        
        
        for block_name, block in blocks_dict.items():

            for key, value in block.items():

                if isinstance(value,list):  

                    main_parameters_values.append(value)
                    main_parameters_names.append(f"{block_name} {key}")
        
                else:

                    extra_parameters_values.append(value)
                    extra_parameters_names.append(f"{block_name} {key}")

        
        main_parameters_values = np.array(main_parameters_values)
        main_parameters_names = np.array(main_parameters_names)
        extra_parameters_values = np.array(extra_parameters_values)
        extra_parameters_names = np.array(extra_parameters_names)
        
        return main_parameters_values, main_parameters_names, extra_parameters_values, extra_parameters_names



    @staticmethod
    def vector_to_dict(main_parameters_values, main_parameters_names, extra_parameters_values, extra_parameters_names):

        blocks_return = {}
        all_parameters = np.concatenate((main_parameters_names, extra_parameters_names))
        all_values = np.concatenate((main_parameters_values, extra_parameters_values))

        for parameter, value in zip(all_parameters, all_values):
            
            blockandparameter = parameter.split()
            blockname = blockandparameter[0]
            parametername = ' '.join(blockandparameter[1:])
            blockcheck = blocks_return.get(blockname)

            if blockcheck:
                blocks_return[blockname][parametername] = value
            else:
                blocks_return[blockname] = {}
                blocks_return[blockname][parametername] = value
        return blocks_return

    @staticmethod
    def complete_dict(parameters_names,parameters_values,blocks_dict):

        blocks_copy = copy.deepcopy(blocks_dict)

        for parameter, value in zip(parameters_names, parameters_values):
            
            blockandparameter = parameter.split()
            blockname = blockandparameter[0]
            parametername = ' '.join(blockandparameter[1:])
            blocks_copy[blockname][parametername] = value
        return blocks_copy



    @staticmethod
    def calc_fixed_parameters2(blocks_dict):
        
        blocks_dict_copy = copy.deepcopy(blocks_dict)


        for blockname, block in blocks_dict.items():
            for key, value in block.items():   
                if callable(block[key]):

                    calculated_value = value(blocks_dict)

                    if isinstance(calculated_value, dict):

                        blocks_dict_copy.update(calculated_value)
                        break

                    else:
                        blocks_dict_copy[blockname][key] = calculated_value

        return blocks_dict_copy


    @staticmethod
    def unnormalize(main_parameters_values, main_parameters_ranges):


        min_values = main_parameters_ranges[:, 0]
        max_values = main_parameters_ranges[:, 1]

        ranges = max_values - min_values
        
        main_parameters_values_copy = main_parameters_values.copy()
        main_parameters_values_copy = main_parameters_values * ranges + min_values
        
        return main_parameters_values_copy
    


    @staticmethod
    def unnormalize_log(main_parameters_values, main_parameters_ranges):


        main_parameters_values_copy = main_parameters_values.copy()

        for i in range(len(main_parameters_values)):

            vmin, vmax = main_parameters_ranges[i]
            
            if vmin < 0 and vmax > 0:
                
                abs_max = max(abs(vmin), abs(vmax))
                log_max = np.log10(abs_max + 1e-10)

                
                signed_val = main_parameters_values[i] - 0.5
                real_val = np.sign(signed_val) * 10**(abs(signed_val) * 2 * log_max)

                main_parameters_values_copy[i] = real_val

            elif vmin > 0:
                
                log_min = np.log10(vmin)
                log_max = np.log10(vmax)

                order = main_parameters_values[i] * (log_max - log_min) + log_min
                main_parameters_values_copy[i] = 10**order

            elif vmax < 0:
                
                log_min = np.log10(abs(vmax))
                log_max = np.log10(abs(vmin))

                order = main_parameters_values[i] * (log_max - log_min) + log_min
                main_parameters_values_copy[i] = -10**order

            else:
                
                main_parameters_values_copy[i] = main_parameters_values[i] * (vmax - vmin) + vmin

        return main_parameters_values_copy
    


    def change_spheno_parameters(self, blocks_dict):


        with open(self.spheno_input_file_path, 'r') as f:
            lines = f.readlines()

        in_block = False
        block_to_change = None
        modified = False

        new_content = []

        for line in lines:

            line_ = line.lstrip()

            if line_.startswith("Block"):

                words = line_.split()
                block_name = words[1]

                if block_name in blocks_dict:

                    in_block = True
                    block_to_change = block_name

                else:

                    in_block = False

                new_content.append(line)
                continue

            if in_block:

                words = line.split()
                index = words.index("#")
                parameter_name = words[index + 1]

                if parameter_name in blocks_dict[block_to_change]:

                    new_value = f"{blocks_dict[block_to_change][parameter_name]:.6E}"
                    words[index - 1] = new_value

                    new_line = f" {' '.join(words[:index])}    # {' '.join(words[(index + 1):])}\n"
                    new_content.append(new_line)

                    modified = True

                else:
                    new_content.append(line)

            else:

                new_content.append(line)

        if modified:

            with open(self.spheno_input_file_path, 'w') as f:
                f.writelines(new_content)



    def run_spheno(self): 


        try:

            input_dir = os.path.dirname(self.spheno_input_file_path)

            command = subprocess.run([self.spheno_file_path, self.spheno_input_file_path],
                                    check=True,
                                    stdout = subprocess.PIPE,
                                    stderr = subprocess.PIPE,
                                    cwd = input_dir)

            output = command.stdout.decode()
            message = output

            spheno_run = True
            
            if "There has been a problem during the run" in output:
                raise RuntimeError("SPheno reportó un fallo en su salida estándar.")
                    
        except RuntimeError as e:
            
            spheno_run = None
            message = str(e)
    
        return spheno_run, message
    


    def run_micromegas(self):
        command = f"ulimit -c 0 && {self.micromegas_file_path} {self.spheno_output_file_path}"
        work_dir = os.path.dirname(self.spheno_output_file_path)
        master, slave = pty.openpty()

        try:
            pid = os.fork()
            if pid == 0:  # Proceso hijo
                os.chdir(work_dir)
                os.dup2(slave, 1)  # Redirige stdout al terminal emulado
                os.dup2(slave, 2)  # Redirige stderr al terminal emulado
                os.close(master)
                os.execvp("sh", ["sh", "-c", command])
            else:  # Proceso padre
                os.close(slave)
                output = b""
                while True:
                    try:
                        chunk = os.read(master, 1024)  # Lee la salida en trozos
                        if not chunk:
                            break
                        output += chunk
                    except OSError:
                        break
                os.waitpid(pid, 0)
        finally:
            os.close(master)

        output_str = output.decode("utf-8")

        def extract_value(pattern, default="0.000E+00"):
            match = re.search(pattern, output_str)
            return match.group(1).replace("e", "E") if match else default

        
        xf = extract_value(r"Xf=([\-\+eE0-9.]+)")
        omega = extract_value(r"Omega h\^2=([\-\+eE0-9.]+)")
        sigma_v = extract_value(r"v⟩\s+=\s+([\-\+eE0-9.]+)")

        
        cdm_nucleon_data = {
            element: {
                "SI": si.replace("e", "E"),
                "SD": sd.replace("e", "E")
            }
            for element, si, sd in re.findall(r"(proton|neutron)\s+SI\s+([\-\+eE0-9.]+)\s+SD\s+([\-\+eE0-9.]+)", output_str)
        }

        
        with open(self.spheno_output_file_path, "a") as spheno_file:
            spheno_file.write("Block MICROMEGAS  # MicrOMEGAs Results\n")
            spheno_file.write(f"    1   {xf}   # Xf (Freeze-out temperature)\n")
            spheno_file.write(f"    2   {omega}   # Omega h^2 (Dark matter relic density)\n")
            for element, values in cdm_nucleon_data.items():
                spheno_file.write(f"    3   {values['SI']}   # {element} SI\n")
                spheno_file.write(f"    4   {values['SD']}   # {element} SD\n")
            spheno_file.write(f"    5   {sigma_v}   # sigma_v [cm^3/s] (indirect detection)\n")
    
    def read_outputs(self, blocks_dict):


        blocks_copy = copy.deepcopy(blocks_dict)

        with open(self.spheno_output_file_path, 'r') as file:
            lines = file.readlines()

        in_block_specified = False
        in_block = False
        block_to_read = None

        for line in lines:

            line_ = line.lstrip()

            if line_.startswith("Block"):

                words = line_.split()
                block_name = words[1]

                if block_name in blocks_dict and isinstance(blocks_dict[block_name],dict):
                    
                    in_block_specified = True
                    block_to_read = block_name

                elif block_name in blocks_dict and callable(blocks_dict[block_name]):

                    block_to_read = block_name
                    blocks_copy[block_name] = {}
                    in_block = True

                else:
                    in_block = False
                    in_block_specified = False

                continue

            elif line_.startswith("DECAY"):

                words = line_.split()
                decay = f"{words[0]} {words[1]}"
                
                if decay in blocks_dict and isinstance(blocks_dict[decay],dict) and blocks_dict[decay].get("WIDTH"):

                    if isinstance(blocks_dict[decay]["WIDTH"],list):

                        blocks_copy[decay]["WIDTH"] = [float(words[2])]

                    else:

                        blocks_copy[decay]["WIDTH"] = float(words[2])

                    in_block_specified = True
                    block_to_read = decay

                elif decay in blocks_dict and isinstance(blocks_dict[decay],dict) :

                    in_block_specified = True
                    block_to_read = decay

                elif decay in blocks_dict and callable(blocks_dict[block_name]):

                    in_block = True
                    blocks_copy[block_name] = {}
                    block_to_read = decay

                else:
                    in_block = False
                    in_block_specified = False

                continue

            if in_block_specified:

                words = line.split()
                parameter_id = words.index("#") + 1
                parameter_name = " ".join(words[parameter_id:])
                
                if parameter_name in blocks_dict[block_to_read]:

                    if isinstance(blocks_dict[block_to_read][parameter_name],list):
                        value_id = next(i for i, elemento in enumerate(words) if "E+" in elemento or "E-" in elemento)
                        blocks_copy[block_to_read][parameter_name] = [(float(words[value_id]))]

                    else:

                        value_id = next(i for i, elemento in enumerate(words) if "E+" in elemento or "E-" in elemento)
                        blocks_copy[block_to_read][parameter_name] = float(words[value_id])

            if in_block:

                words = line.split()
                parameter_id = words.index("#") + 1
                parameter_name = " ".join(words[parameter_id:])
                value_id = next(i for i, elemento in enumerate(words) if "E+" in elemento or "E-" in elemento)
                blocks_copy[block_to_read][parameter_name] = float(words[value_id])

        return blocks_copy



    def log_likelihood(self, expected_data_dict, model_data_dict):

        model_values, _, extra_model_values,_ = self.dict_to_vector(model_data_dict)

        extra_model_values = extra_model_values.astype(float)

        blob = np.concatenate((model_values[:,0], extra_model_values))
        
        if self.dinamic_likelihood == True:

            new_expected_data_dict = {}  

            for blockname, block in expected_data_dict.items():
                for key, value in block.items():   
                    if callable(block[key]):

                        calculated_value = value(expected_data_dict)

                        if isinstance(calculated_value, dict):

                            new_expected_data_dict.update(calculated_value)
                            break

                        else:
                            new_expected_data_dict[blockname][key] = calculated_value
        else:
            new_expected_data_dict = expected_data_dict 

        expected_values = self.dict_to_vector(new_expected_data_dict)[0]
        
        loglikelihood = -0.5 * np.sum(((model_values[:,0] - expected_values[:,0]) / expected_values[:,1]) ** 2)

        

        return loglikelihood, blob



    def log_prior_before_SPheno(self, main_parameters_values, main_parameters_ranges, blocks_dict):


        if isinstance(self.constraints_before_spheno, list) and len(self.constraints_before_spheno) > 0:
            
            for i in range(len(self.constraints_before_spheno)):

                if not(self.constraints_before_spheno[i](blocks_dict)):

                    return -np.inf

        min_vals = main_parameters_ranges[:, 0]
        max_vals = main_parameters_ranges[:, 1]


        if ((self.strict_parameters_ranges == True ) 
            and (np.any(main_parameters_values < min_vals) 
                 or np.any(main_parameters_values > max_vals))):
            
            return -np.inf
        
        return 0.0
    


    def log_prior_after_SPheno(self, model_data_dict):


        if isinstance(self.constraints_after_spheno, list) and len(self.constraints_after_spheno) > 0:

            for i in range(len(self.constraints_after_spheno)):

                if not(self.constraints_after_spheno[i](model_data_dict)):

                    return -np.inf


        return 0.0
    


    def log_posterior(self, main_parameters_values):


        if self.log_parameterization == True:

            normalized_main_parameters_values = self.unnormalize_log(main_parameters_values, self.main_parameters_ranges)

        else:

            normalized_main_parameters_values = self.unnormalize(main_parameters_values, self.main_parameters_ranges)

        spheno_input_blocks_dict = self.complete_dict(self.main_parameters_names, normalized_main_parameters_values, self.spheno_block_dict)
            
        complete_spheno_input_dict = self.calc_fixed_parameters2(spheno_input_blocks_dict)
        full_parameters_values = self.dict_to_vector(complete_spheno_input_dict)[2]
        prior_bs = self.log_prior_before_SPheno(normalized_main_parameters_values,self.main_parameters_ranges, complete_spheno_input_dict)

        if np.isinf(prior_bs):

            return -np.inf, np.append(full_parameters_values,np.full(len(self.blob_names)-len(full_parameters_values)+1, 0))
            
        self.change_spheno_parameters(complete_spheno_input_dict)

        spheno_out = self.run_spheno()[0]
        if spheno_out is None:

            return None
        
        if self.use_micromegas == True:

            self.run_micromegas()

        calculated_model_data_dict = self.read_outputs(self.expected_data_dict)

        prior_as = self.log_prior_after_SPheno(calculated_model_data_dict | complete_spheno_input_dict)

        if np.isinf(prior_as):
            model_values, _, extra_model_values,_ = self.dict_to_vector(calculated_model_data_dict)

            extra_model_values = extra_model_values.astype(float)

            blob = np.concatenate((model_values[:,0], extra_model_values))
            full_blob = np.concatenate((full_parameters_values, blob))
            return -np.inf, np.concatenate((full_parameters_values, blob, np.array([0])))
        #This does not show the points that are rejectedbefore SPheno in the output csv because of the
        #conditional that is below in the run. I want to change it to have 3 conditions. To save everything,
        #everything accepted before SPheno, and just points accepted after all

        log_lik, blob = self.log_likelihood(self.expected_data_dict,calculated_model_data_dict)
        
        posterior = prior_bs + prior_as + log_lik
        full_blob = np.concatenate((full_parameters_values, blob))
        full_blob = np.append(full_blob, np.exp(posterior))

        return posterior, full_blob



    def run(self):


        def log_posterior_caller(params):

            return self.log_posterior(params)

        
        self.main_parameters_ranges, self.main_parameters_names, _ , _ = self.dict_to_vector(self.spheno_block_dict)

        ndim = self.main_parameters_ranges.shape[0]
        nw = self.nwalkers
        a_ = self.stretch

        init_parameters = np.random.uniform(0, 1, (nw, ndim))

        spheno_out = self.run_spheno()[0]
        if spheno_out is None:

            return None
        
        complete_spheno_block_dict = self.complete_dict(self.main_parameters_names, init_parameters[0,:], self.spheno_block_dict)

        complete_spheno_block_dict = self.calc_fixed_parameters2(complete_spheno_block_dict)

        calculated_output_data_dict = self.read_outputs(self.expected_data_dict)

        complete_spheno_block_names = self.dict_to_vector(complete_spheno_block_dict)[3]

        _, calculated_output_data_names, _, calculated_output_data_names2 = self.dict_to_vector(calculated_output_data_dict)

        self.blob_names = np.concatenate([complete_spheno_block_names, calculated_output_data_names, calculated_output_data_names2])


        self.sampler = emcee.EnsembleSampler(nw,
                                        ndim, 
                                        log_posterior_caller,
                                        blobs_dtype=object, a = a_)



        accepted_samples = 0 

        with open(self.output_mcmc_file, mode="w", newline="", encoding="utf-8") as file:

            f = csv.writer(file, delimiter=";")
            header = list(np.append(self.blob_names, "likelihood").astype(str))
            f.writerow(header)
            

            with tqdm(total=self.accepted_points) as pbar:

                for sample in self.sampler.sample(init_parameters, iterations = self.steps, progress = True, store = False):

                    log_prob = sample.log_prob
                    accepted_samples += len(log_prob[log_prob > self.likelihood_threshold])
                    pbar.update(len(log_prob[log_prob > self.likelihood_threshold]))

                    row = sample.blobs.astype(float)
                    if self.write_only_accepted == False:
                        row = row[row[:,-1] >= 0]
                        row = np.unique(row, axis=0)
                        f.writerows(row)

                    else:

                        accepted_samples_array = row[row[:,-1] >= np.exp(self.likelihood_threshold)]
                        accepted_samples_array = np.unique(accepted_samples_array, axis=0)
                        f.writerows(accepted_samples_array)

                    if accepted_samples >= self.accepted_points:
                        print(f"¡Criterio alcanzado! {accepted_samples} muestras aceptadas.")
                        break
                        


    def get_spheno(self, header, row):


        blocks_dict = copy.deepcopy(self.spheno_block_dict)
        row = row.astype(float)
        data = dict(zip(header, row))

        for param, value in data.items():
            for category, param_dict in blocks_dict.items():

                if param in param_dict:
                    param_dict[param] = value


        self.change_spheno_parameters(blocks_dict)

        self.run_spheno()

        return None
    