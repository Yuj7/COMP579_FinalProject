import pandas as pd

class Logger:
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

            filename = f'./logs/PPO_case{self.case}.txt'
            with open(filename, 'w') as f:
                f.write(df.to_string())
