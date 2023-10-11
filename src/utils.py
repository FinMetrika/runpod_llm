import sys
import logging



def update_config(FLAGS):
    """
    Update config arguments if any change was done via CLI when
    running "sh run.sh". FLAGS argument is instantiation of the
    Config dataclass.
    """
    for i in range(1, len(sys.argv),2):
        attr_name = sys.argv[i]
        attr_value = sys.argv[i+1]
    
        if hasattr(FLAGS, attr_name):
            setattr(FLAGS, attr_name, attr_value)
        else:
            logging.warning(F'No such attribute: {attr_name}')