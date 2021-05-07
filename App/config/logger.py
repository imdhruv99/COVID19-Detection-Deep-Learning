from datetime import datetime
datetime.datetime.now()

import logger



class Logger:

    def __init__(self,run_id,log_module,log_file_name):
        self.logger.setLevel(logging.DEBUG)
        if log_file_name=='training':
            file_handler = logging.FileHandler('logs/training_logs/train_log_' +  + '.log')
        else:
            file_handler = logging.FileHandler('logs/prediction_logs/predict_log_'  '.log')

        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self,message):
        self.logger.info(message)

    def exception(self,message):
        self.logger.exception(message)