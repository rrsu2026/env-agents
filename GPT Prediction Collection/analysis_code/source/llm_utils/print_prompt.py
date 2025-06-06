import sys
# setting path
sys.path.append('../')

import json
import numpy
import datetime
import random

from source.llm_utils.settings import * 
from source.global_methods import *
from source.llm_utils.gpt_structure import *

##############################################################################
#                    PERSONA Chapter 1: Prompt Structures                    #
##############################################################################

def print_run_prompts(prompt_template=None, 
                      prompt_input=None,
                      prompt=None, 
                      output=None): 
  # print (f"=== START =======================================================")
  # print ("~~~ prompt_template    -------------------------------------------")
  # print (f"=== {prompt_template}")
  # print ("~~~ prompt_input    ----------------------------------------------")
  # print (prompt_input, "\n")
  # print ("~~~ prompt    ----------------------------------------------------")
  # print (prompt, "\n")
  # print ("~~~ output    ----------------------------------------------------")
  # print (output, "\n") 
  # print ("=== END ==========================================================")
  # print ("\n\n\n")
  print ("")
