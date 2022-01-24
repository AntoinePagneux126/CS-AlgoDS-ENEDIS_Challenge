import sys, os , inspect

#############################################
#####   Import gestion across project   #####
#############################################

current_dir= os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir= os.path.dirname(current_dir)
sys.path.insert(0,parent_dir)

import code
from code.merge_inout import merge
import pytest

def test_loader():
    
    with pytest.raises(Exception) as execinfo : 
        merge(file_in=3)
        assert(str(execinfo.value)=="file must be string which represent path to csv files")
    with pytest.raises(Exception) as execinfo : 
        merge(file_out=3)
        assert(str(execinfo.value)=="file must be string which represent path to csv files")
    