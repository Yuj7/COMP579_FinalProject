import pandas as pd

class Logger:
    def __init__(self):
        pass
    def log_to_file(self, data : list[list], filename : str, columns : list[str]) -> None:
        """
        Logs data into a txt file
        
        Args:
            data (list[list]): List of the lists representing the columns to log
            filename (str): Name of the file
            columns (list[str]): name of columns
        """
        data_dict = dict(zip(columns, data))
        df = pd.DataFrame(data_dict)

        with open(f'./logs/{filename}.txt', 'w') as f:
            f.write(df.to_string())

    class PPO:
        def __init__(self, case):
            self.case = case
            self.log = {
                'actor_loss' : [],
                'value_loss' : [],
                'advantages' : []
            }

        def log_training_metrics(self):
            df = pd.DataFrame(self.log)

            filename = f'./logs/PPO_case{self.case}_TrainingData.txt'
            with open(filename, 'w') as f:
                f.write(df.to_string())

