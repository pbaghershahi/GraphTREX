from os.path import dirname, join, realpath

# Basic Constants
BASE_PATH = dirname(realpath(__file__))
BASIC_CONF_PATH = join(BASE_PATH, 'configs/basic.conf')
PRETRAINED_MODEL = None 
USE_UMLS=False# Datasets Constants
NOT_ENTITY = 'not-entity'
NOT_RELATION = 'not-relation'
SPLITS = {'train', 'dev', 'test'}

E3C ='e3c'
E3C_ENTITY_TYPES = [NOT_ENTITY, 'EVENT']#, 'TIMEX']#, 'PATIENT', 'H-PROFESSIONAL', 'OTHER']#, 'TIME']
E3C_RELATION_TYPES = [NOT_RELATION, 'BEFORE', 'OVERLAP']#  'SIMULTANEOUS', 'ENDS-ON', 'BEGINS-ON', ]
         
TEMPORAL = 'i2b2'

EXACT = True

TEMPORAL_ENTITY_TYPES = [NOT_ENTITY, 'ADMISSION', 'DISCHARGE','TREATMENT',
                          'PROBLEM',  'TEST', 'CLINICAL_DEPT',  'EVIDENTIAL',
                            'DATE', 'OCCURRENCE', 'TIME', 'DURATION', 'FREQUENCY']


TEMPORAL_RELATION_TYPES = [NOT_RELATION, "BEFORE", "AFTER", "OVERLAP"]# -1, 1, 0]

DATASETS = [TEMPORAL, E3C]
