from optparse import OptionParser
import importlib

parser = OptionParser()
parser.add_option('--config', dest='cli_config', default=None, type='str')
(options,args) = parser.parse_args()
cli_config = options.cli_config
cli_config = importlib.import_module(cli_config)
conf = cli_config.models_genesis_config()
conf.display()