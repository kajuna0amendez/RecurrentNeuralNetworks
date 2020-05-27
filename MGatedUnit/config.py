# -*- coding: utf-8 -*-
#!/usr/bin/env python
__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2019"
__credits__ = ["Andres Mendez-Vazquez"]
__license__ = "Open"
__version__ = "v1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email =  "kajuna0kajuna@gmail.com"
__status__ = "Development"

import numpy as np
import time

########################### Information about Model ############################
# Training Epochs
num_epochs = 50000
# Minbatch
minbatch = 60
# individual series lenght
T_slen = 100
# Output Size
output_dim = 1
iter_value = 100
millis = np.uint32(round(time.time() * 1000))



####################### Parameters for ADAM ####################################
minbatch_alpha= 0.01#0.1 0.001
adam_alpha_config = 0.0001 #0.001 #0.0001
epsilon_adam = 1e-4
Beta1_config = 0.7 #0.7
Beta2_config = 0.9 #0.9


########################### Information about Data #############################

# Series Lenght
series_length = 500