import subprocess
import numpy as np
import copy
import emcee
import os
import pty
import re
import csv
from tqdm import tqdm

#incluyendo la verosimilitud en función
class smmcmc:
    def __init__(self, SphenoBlocksDict, ExpectedDataDict, ConstraintsBeforeSPheno, ConstraintsAfterSPheno,
                SPhenoFilePath, SPhenoInputFilePath, SPhenoOutputFilePath, UseMicrOmegas = False, MicrOmegasFilePath = " ",
                nWalkers = 100, LikelihoodThreshold = 0.9, AcceptedPoints = 1000, OutputMCMCfile = "output_mcmc.csv",
                Steps = None, Stretch = 1.2, LogParameterization = False, StrictParametersRanges = True, WriteOnlyAccepted = True):

        """
        La clase `smmcmc` permite explorar el espacio de parámetros de un modelo escotogénico utilizando el método de Markov Chain Monte Carlo (MCMC) 
        y las herramientas computacionales SPheno y, opcionalmente, MicrOmegas. SPheno es indispensable para realizar los cálculos relacionados 
        con las masas, los branching ratios y otros parámetros relevantes, mientras que MicrOmegas se emplea, 
        si se activa, para estudiar propiedades de la materia oscura, como la densidad relicta. 

        Para el correcto funcionamiento de esta clase, es fundamental que el modelo físico esté completamente implementado y funcional tanto en SPheno 
        como, si aplica, en MicrOmegas. Esto incluye garantizar que las configuraciones y definiciones de entrada estén correctamente configuradas 
        en ambos softwares.

        ### Argumentos:
        
        ### SphenoBlocksDict (dict):
        Este argumento es un diccionario que contiene la definición de los parámetros del modelo que serán explorados durante el análisis. 
        Es un diccionario de diccionarios estructurado en bloques que corresponden a los bloques de entrada del archivo de input de SPheno. 
        Cada bloque se representa mediante un diccionario interno, donde:

        - Las *llaves* corresponden a los nombres de los bloques, tal como se definen en el archivo de entrada de SPheno (por ejemplo de BLOCK MINPAR, `"MINPAR"`).
        - Los *valores* dentro de cada bloque son los parámetros que se quieren explorar. Estos parámetros pueden ser definidos de tres maneras diferentes:

            1. **Lista:** Si el parámetro debe variar dentro de un intervalo, se especifica una lista con dos elementos: 
               el valor mínimo y el máximo del intervalo. Si se desea restringir estrictamente los valores generados al rango especificado, 
               puede activarse el argumento `StrictParametersRanges`.

            2. **Número flotante (float):** Si el parámetro debe permanecer constante durante el análisis, se define directamente 
               con su valor numérico.

            3. **Función:** Si el valor de un parámetro debe calcularse dinámicamente en función de otros parámetros, se define una función.
            Esta función debe ser definida previamente y debe tomar como argumento un diccionario que contiene todos los parámetros del modelo, accesibles mediante 
            sus nombres especificando el bloque. La función debe devolver un único valor flotante. Por ejemplo:

               ```python
               def lambda1_fixed(parameters: dict):
                    other = parameters["OTHER"]
                    lambda_1 = (other["mH1"]**2) * (np.cos(other["alpha"])**2)
                    lambda_1 += (other["mH2"]**2) * (np.sin(other["alpha"])**2)
                    lambda_1 = lambda_1/(other["v"]**2)
                    return lambda_1
               ```
            
            Si se quiere calcular varios parámetros a la vez con una sola función, al menos uno de los parámetros dentro de un bloque debe estar definido
            como una función. No se puede definir directamente el nombre de un bloque como una función, ya que esto generaría un error. En su lugar, se debe nombrar un 
            parámetro dentro del bloque como una función, y esta debe retornar un diccionario que contenga los bloques y los parámetros que se desean calcular. Al igual que
            en el caso de un único parámetro, esta función debe ser definida previamente y debe tomar como argumento un diccionario que contiene todos los parámetros 
            del modelo, accesibles mediante sus nombres especificando el bloque. Por ejemplo:
               
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

        Además, es posible incluir parámetros adicionales que no están incluídos en los bloques de SPheno pero que pueden ser necesarios para los cálculos. 
        Estos parámetros pueden agruparse en bloques personalizados con nombres arbitrarios (por ejemplo, `"OTHER"`) para organizar valores fijos o generados 
        que serán utilizados en funciones.

        #### Ejemplo de `SphenoBlocksDict`:
        A continuación, se muestra un ejemplo de diccionario correctamente estructurado para este argumento:

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

        Este ejemplo incluye dos bloques principales: `"MINPAR"`, que contiene parámetros directamente utilizados en SPheno, y `"OTHER"`, 
        que define parámetros adicionales necesarios para cálculos específicos (como constantes o valores de entrada para funciones dependientes).

        #### Consideraciones adicionales:

        - Los nombres de los parámetros deben coincidir exactamente con los definidos en los archivos de entrada de SPheno. 
        - Las funciones utilizadas para calcular parámetros dinámicos deben estar correctamente documentadas y validadas, ya que errores 
          en su definición pueden generar resultados inesperados durante la ejecución del análisis.
       
        ### ExpectedDataDict (dict):

        Este argumento es un diccionario que define las cantidades a utilizar para evaluar la verosimilitud de los parámetros en el análisis. Se emplea para leer datos 
        específicos de los archivos de salida generados por SPheno, y opcionalmente por MicrOmegas, a fin de calcular la verosimilitud de los parámetros en cada iteración 
        del proceso MCMC. Además, permite extraer otras cantidades relevantes para realizar análisis adicionales o aplicar restricciones.

        La estructura de este diccionario es similar a la de 'SphenoBlocksDict', ya que se organiza en bloques. Cada bloque está representado por un diccionario interno, estructurado de la siguiente manera:

        1. **Llaves del diccionario principal:** 
            Las llaves corresponden a los nombres de los bloques en los archivos de salida de SPheno o MicrOmegas. Por ejemplo:
            - En el caso de SPheno, un bloque puede ser `"MASS"` (equivalente al bloque `BLOCK MASS` en el archivo de salida).
            - Para decaimientos, se utiliza `"DECAY No"`, donde `No` es el identificador numérico del decaimiento.
            - Si se incluye MicrOmegas, se debe especificar un bloque llamado `"MICROMEGAS"`.

        2. **Diccionarios internos (valores asociados a cada bloque):**
            Cada bloque contiene un diccionario con los siguientes elementos:
            - Las *llaves* representan las cantidades de interés.
            - Los *valores* pueden adoptar dos formatos distintos dependiendo del propósito:
                - **Lista:** Si la cantidad se usará para el cálculo de la verosimilitud, el valor debe ser una lista de dos elementos: 
                    1. El valor esperado (extraído de literatura o experimentos).
                    2. Su incertidumbre asociada.
                - **Cadena de texto (str):** Si la cantidad no contribuye directamente a la verosimilitud pero se requiere para análisis adicionales, el valor debe ser una cadena. 
                Se recomienda que esta cadena represente un número (por ejemplo, `"0.0"`) para garantizar que, en caso de no encontrarse el valor en el archivo de salida, se asignará dicho número por defecto.
                - **Función:** Se pueden definir funciones para calcular dinámicamente los valores de la verosimilitud. Esto permite evaluar la verosimilitud en cada iteración 
                del proceso MCMC sin necesidad de fijar valores estáticos. Esta función debe ser definida previamente y debe tomar como argumento un diccionario que contiene todos 
                los parámetros que se incluyan en este diccionario (que se leeran de SPheno o Micromegas), accesibles mediante sus nombres especificando el bloque.
                    1.Para calcular una única cantidad, la función debe devolver una lista de dos elementos [valor esperado, incertidumbre].
                    2.Para calcular varias cantidades a la vez, la función debe devolver un diccionario donde las llaves sean las cantidades y los valores sean:
                        a.Una lista [valor esperado, incertidumbre] si la cantidad contribuye a la verosimilitud.
                        b.Un valor flotante si solo se desea extraer para análisis adicionales.

                    Por ejemplo:           
                    def dynamic_quantities(parameters: dict):
                        return {
                            "Mass_H1": [parameters["mH1"], 0.05 * parameters["mH1"]],
                            "Mass_H2": [parameters["mH2"], 0.05 * parameters["mH2"]],
                            "Mixing_Angle": parameters["alpha"]
                        }

        #### Detalles adicionales:
        - Todos los valores especificados en este diccionario que estén presentes en los outputs de SPheno o MicrOmegas se incluirán en el archivo de salida generado por el análisis. Esto es útil para almacenar 
        información adicional que no se utiliza en el cálculo de la verosimilitud pero puede ser relevante para otros análisis.
        - Para decaimientos, los bloques deben incluir el prefijo `"DECAY "` seguido del identificador de la partícula en cuestión. Por ejemplo:
        ```python
        "DECAY 25": {
            "WIDTH": [3.7e-3, 1.7e-3],
            "BR(hh_1 -> Ah_2 Ah_2)": "0.0",
            "BR(hh_1 -> FX0_1 FX0_1)": "0.0",
        }
        ```

        Si se utiliza MicrOmegas, debe incluirse un bloque llamado "MICROMEGAS" con las cantidades específicas a extraer. Por ejemplo:
        
        ```python
        "MICROMEGAS":
        {
            "Xf (Freeze-out temperature)": "0.0",
            "Omega h^2 (Dark matter relic density)": "0.0" 
        }
        ```

        #### Aquí un ejemplo de `ExpectedDataDict` bien construido:
        ```python
        ExpectedDataDict = { 
        "MASS":
        {
            "hh_1" : "0.0",
            "FX0_1" : "0.0", 

        },
        "DECAY 25":
        {   
            "WIDTH": [3.7e-3,1.7e-3],
            "BR(hh_1 -> Ah_2 Ah_2 )": "0.0",
            "BR(hh_1 -> FX0_1 FX0_1 )" : "0.0",
        },
        "MICROMEGAS":
        {
            "Xf (Freeze-out temperature)": "0.0",
            "Omega h^2 (Dark matter relic density)": [0.120, 0.0036]
        }
        }
        ```

        ### ConstraintsBeforeSPheno (list):

        Este argumento permite definir restricciones (`constraints`) que deben verificarse antes de ejecutar SPheno en cada iteración del análisis. Estas restricciones se aplican a los parámetros 
        que se encuentran en `SphenoBlocksDict`. Implementar los `constraints` de esta manera resulta en un análisis más eficiente, ya que evita correr SPheno y/o MicrOmegas en 
        iteraciones donde los parámetros de entrada ya no cumplen con las restricciones definidas.

        #### Definición:
        `ConstraintsBeforeSPheno` es una lista de funciones previamente definidas. Cada función debe cumplir con las siguientes características:
        1. **Recibir un único argumento:** 
        Un diccionario que sigue la misma estructura que `SphenoBlocksDict`, no se puede incluir elementos que no se hayan nombrado previamente ahí. 
        Este diccionario representa los bloques y parámetros que se usarán para evaluar las restricciones.
        2. **Retornar un valor booleano:**
        - Retorna `True` si los parámetros cumplen con la restricción evaluada.
        - Retorna `False` si no se cumple la restricción.

        #### Ejemplo:
        A continuación, se muestra un ejemplo detallado de cómo definir y usar un `ConstraintsBeforeSPheno`:

        
        ####Definición de una función de restricción
        ```python
        def chi_masses(blocks):
            # Extracción de los valores del bloque KAPPAIN
            kappain = blocks["KAPPAIN"]
            chi_1 = kappain["Kap(1,1)"]
            chi_2 = kappain["Kap(2,2)"]
            chi_3 = kappain["Kap(3,3)"]

            # Evaluación del constraint
            return (chi_1 < chi_2 and chi_2 < chi_3)
        ```

        #### Definición de ConstraintsBeforeSPheno
        ```python
        ConstraintsBeforeSPheno = [chi_masses]
        ```

        ### ConstraintsAfterSPheno (list):

        Este argumento permite definir restricciones (`constraints`) que deben verificarse después de ejecutar SPheno en cada iteración del análisis. Estas restricciones se aplican a los parámetros 
        que se encuentran en `ExpectedDataDict` y/o `SphenoBlocksDict`. 
        #### Definición:
        `ConstraintsAfterSPheno` es una lista de funciones previamente definidas. Cada función debe cumplir con las siguientes características:
        1. **Recibir un único argumento:** 
        Un diccionario que sigue la misma estructura que `ExpectedDataDict` y/o `SphenoBlocksDict`, no se puede incluir elementos que no se hayan nombrado previamente ahí. 
        Este diccionario representa los bloques y parámetros que se usarán para evaluar las restricciones.
        2. **Retornar un valor booleano:**
        - Retorna `True` si los parámetros cumplen con la restricción evaluada.
        - Retorna `False` si no se cumple la restricción.

        #### Ejemplo:
        A continuación, se muestra un ejemplo detallado de cómo definir y usar un `ConstraintsAfterSPheno`:

        
        ####Definición de una función de restricción
        ```python
        def branching_ratios(model_decay:dict):
            alpha = model_decay["OTHER"]["alpha"]
            model_decay_ = model_decay["DECAY 25"]
            br1 = float(model_decay_["BR(hh_1 -> Ah_2 Ah_2 )"])
            br2 = float(model_decay_["BR(hh_1 -> FX0_1 FX0_1 )"])
            br3 = float(model_decay_["BR(hh_2 -> FX0_2 FX0_2 )"])
            br4 = float(model_decay_["BR(hh_2 -> FX0_3 FX0_3 )" ])
    
            return ((np.cos(alpha)**2) *(br1 + br2 + br3 +br4)) < 0.19
        ```

        #### Definición de ConstraintsAfterSPheno
        ```python
        ConstraintsAfterSPheno = [branching_ratios]
        ```
        
        ### SPhenoFilePath (string):

        Este parámetro indica la ruta absoluta al archivo ejecutable del modelo de SPheno.

        Ejemplo:
        ```python
        SPhenoFilePath = "/home/f.herreras/SPheno-4.0.5/bin/SPhenoScotogenic_SLNV"
        ```

        ### SPhenoInputFilePath (string):

        Este parámetro indica la ruta absoluta al archivo de input del modelo de SPheno.

        Ejemplo:
        ```python
        SPhenoInputFilePath = "/home/f.herreras/SPheno-4.0.5/Scoto_SLNV_11.10/Input_Files/LesHouches.in.Scotogenic_SLNV"
        ```

        ### SPhenoOutputFilePath (string):

        Este parámetro indica la ruta absoluta al archivo de output del modelo de SPheno. Esta es la ruta a la carpeta donde se encuentra el código
        +  SPheno.spc. + el nombre del modelo:

        [Directorio]/SPheno.spc.[Nombre_del_Modelo]

        Ejemplo:
        ```python
        SPhenoOutputFilePath = "/home/f.herreras/Code/SPheno.spc.Scotogenic_SLNV"
        ```

        ### UseMicrOmegas (boolean):

        Este parámetro indica si se usa o no MicrOmegas en el análisis. Por defecto es False, pero si se especifica como True el código llama a MicrOmegas en cada ciclo alimentándolo
        con el output de SPheno del ciclo.

        ### MicrOmegasFilePath (string):

        De haber especificado el uso de MicrOmegas, debe indicarse en este parámetro la ruta absoluta al ejecutable del modelo en MicrOmegas.

        Ejemplo:
        ```python
        MicrOmegasFilePath="/home/f.herreras/micromegas_6.0.5/Scoto_SLNV_11.10/CalcOmega_MOv5"
        ```
        ### nWalkers (int):

        Indica el número de walkers que tendrá el ensemble sampling del Markov Chain Monte Carlo

        ### LikelihoodThreshold (float):

        Indica la verosimilitud mínima que deben tener los parámetros para contar como punto aceptado (no confundir con la probabilidad de aceptación de cada grupo de parámetros)

        ### AcceptedPoints (int):

        Indica la cantidad de puntos aceptados requeridos para finalizar el análisis.

        ### OutputMCMCfile (string):

        Indica el nombre del archivo con los puntos que retornará el análisis.

        ### Steps (False or int):

        Si además de querer un número máximo de puntos aceptados se requiere un número de ciclos máximos se debe especificar el número, si no, dejar en `False`.

        ### Stretch (float):

        En el método de ensemble sampling que utiliza emcee, el parámetro stretch factor es un valor que controla el tamaño de los pasos propuestos por el muestreo 
        en el espacio de parámetros. 

        En el Stretch Move, cada walker propone un nuevo punto basado en la posición de los demás caminantes. 
        El nuevo punto se genera de la siguiente forma:
            y = x + Z(x' - x)

        Donde:
        - x es la posición actual del caminante.
        - x' es una posición elegida al azar de otro caminante en el ensemble.
        - Z es un factor de escala aleatorio que depende del stretch factor a, y se define como:
            Z = Uniform[1/a, a]
        Este Z determina cuánto se "estira" o "contrae" el movimiento hacia el otro 
        caminante, de ahí el nombre Stretch Move.

        - Valores pequeños de Stretch: Producen pasos pequeños (menos exploración del espacio de parámetros, más local).
        - Valores grandes de Stretch: Producen pasos grandes (mayor exploración, pero pueden ser ineficientes si el espacio de parámetros es complicado).

        ### LogParameterization (boolean):

        Al usar emcee, es recomendable trabajar con números en el intervalo [0, 1], lo que implica realizar un reescalado de los datos. Este reescalado 
        puede ser de dos tipos: uniforme (si se indica `False`) o logarítmico (si se indica `True`). La elección entre estos dos métodos depende
        de las ventajas específicas que cada uno ofrezca según el contexto del problema, siendo ambos enfoques válidos dependiendo del caso.

        ### StrictParametersRanges (boolean):

        Indica si los parámetros de `SphenoBlocksDict` definidos como intervalos deben respetar estrictamente sus límites.  
        - Si se establece en `False`, los parámetros pueden salir de los límites del intervalo durante la exploración.  
        - Si se establece en `True`, los parámetros permanecerán estrictamente dentro de los límites definidos.  

        ### WriteOnlyAccepted (boolean):

        Determina qué datos se guardan en el archivo de salida:  
        - Si se establece en `True`, solo se guardan los datos aceptados cuyo valor sea mayor que el umbral definido por `LikelihoodThreshold`.  
        - Si se establece en `False`, se guardan todos los datos explorados en el MCMC, independientemente de su valor.  

        """
        self.spheno_block_dict = SphenoBlocksDict  
        self.expected_data_dict = ExpectedDataDict
        self.constraints_before_spheno = ConstraintsBeforeSPheno 
        self.constraints_after_spheno = ConstraintsAfterSPheno 
        self.spheno_file_path = SPhenoFilePath 
        self.spheno_input_file_path = SPhenoInputFilePath 
        self.spheno_output_file_path = SPhenoOutputFilePath 
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
        for blockname, block in self.expected_data_dict.items():
            for key, value in block.items():   
                if callable(block[key]):
                    self.dinamic_likelihood = True
                    

    
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

            max_values = main_parameters_ranges[i, 1]
            min_values = main_parameters_ranges[i, 0]
            
            if min_values != 0:

                max_values = np.log10(max_values)
                min_values = np.log10(min_values)

                order = main_parameters_values_copy[i]*(max_values - min_values) + min_values

                main_parameters_values_copy[i] = 10**(order)

            else:

                main_parameters_values_copy[i] = main_parameters_values_copy[i]*(max_values - min_values) + min_values

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

            command = subprocess.run([self.spheno_file_path, self.spheno_input_file_path],check=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE )

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
        master, slave = pty.openpty()

        try:
            pid = os.fork()
            if pid == 0:  # Proceso hijo
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

        prior_bs = self.log_prior_before_SPheno(normalized_main_parameters_values,self.main_parameters_ranges, complete_spheno_input_dict)

        if np.isinf(prior_bs):

            return -np.inf, np.append(np.full(len(self.blob_names), 0), -np.inf)
            
        self.change_spheno_parameters(complete_spheno_input_dict)

        spheno_out = self.run_spheno()[0]
        if spheno_out is None:

            return None
        
        if self.use_micromegas == True:

            self.run_micromegas()

        calculated_model_data_dict = self.read_outputs(self.expected_data_dict)

        prior_as = self.log_prior_after_SPheno(calculated_model_data_dict | complete_spheno_input_dict)

        if np.isinf(prior_bs):

            return -np.inf, np.append(np.full(len(self.blob_names), 0), -np.inf)
        

        log_lik, blob = self.log_likelihood(self.expected_data_dict,calculated_model_data_dict)
        full_parameters_values = self.dict_to_vector(complete_spheno_input_dict)[2]
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
                        f.writerows(row)

                    else:

                        accepted_samples_array = row[row[:,-1] > np.exp(self.likelihood_threshold)]
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
    