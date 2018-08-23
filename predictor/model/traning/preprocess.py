import pandas as pd
import requests
import numpy as np

from sklearn import preprocessing


class DataProcessor:
    def __init__(self):
        self.raw_data = None
        self.data = None

    def download(self, url, token=None, token_type='Bearer'):
        response = requests.get(url=url,
                                headers={'Authorization': token_type + ' ' + token},
                                stream=True)

        with open('raw_dataset.csv', 'wb') as file:
            for chunk in response.iter_content(1024):
                if chunk:
                    file.write(chunk)

        self.raw_data = pd.read_csv('raw_dataset.csv')
        df = pd.DataFrame(self.raw_data)

        df['month'] = df['transfer_time'].map(lambda time: int(time.split('-')[1]))
        df['year'] = df['transfer_time'].map(lambda time: int(time.split('-')[0]))

        df.sort_values(by=['customer_id', 'transfer_time'], inplace=True)
        df.drop(['transfer_method', 'category', 'transfer_time'], inplace=True, axis=1)
        net_spending = df.groupby(['customer_id', 'year', 'month'])['balance_diff'].sum()

        new_df = pd.DataFrame(columns=['customer_id', 'occupation', 'birth_year', 'month', 'balance'])

        customer_id = None
        month = None
        for _, row in df.iterrows():
            if row['customer_id'] != customer_id or row['month'] != month:
                new_df = new_df.append({
                    'customer_id': row['customer_id'],
                    'occupation': row['occupation'],
                    'birth_year': row['birth_year'],
                    'month': row['month'],
                    'balance': row['balance']
                }, ignore_index=True)
                customer_id = row['customer_id']
                month = row['month']

        new_df['net_spending'] = list(net_spending)
        self.data = new_df

    def process(self, occupations=None):
        if occupations is None:
            occupations = [
                'MISC',
                'PROFESSIONAL',
                'MANAGERIAL',
                'CLERICAL',
                'MILITARY',
                'ELEMENTARY',
                'TECHNICAL',
                'SERVICE',
                'AGRICULTURAL',
            ]

        df = pd.DataFrame(self.data)
        label_enc = preprocessing.LabelEncoder()
        ohe = preprocessing.OneHotEncoder(sparse=False)
        occupations_enc = np.array(label_enc.fit_transform(occupations))

        # One hot occupation
        df['occupation_enc'] = label_enc.transform(df['occupation'])
        ohe.fit(occupations_enc.reshape(-1, 1))

        occupations_oh = ohe.transform(np.array(df['occupation_enc']).reshape(-1, 1))

        add_df = pd.DataFrame(occupations_oh, columns=occupations)
        result_df = pd.concat([df, add_df], axis=1)

        # One hot month
        months = np.array([month for month in range(1, 13)])
        ohe.fit(months.reshape(-1, 1))

        months_oh = ohe.transform(np.array(result_df['month']).reshape(-1, 1))
        new_df = pd.DataFrame(months_oh, columns=months)
        result_df = pd.concat([result_df, new_df], axis=1)

        result_df.drop(['customer_id', 'occupation_enc', 'occupation', 'month'], inplace=True, axis=1)

        self.data = result_df

        return result_df

    def normalize(self):
        data = np.array(self.data)

        non_cate = data[:, [0, 1, 2]]
        non_cate = preprocessing.normalize(non_cate, axis=0)

        normalized_data = np.concatenate((non_cate, data[:, [col for col in range(3, 22)]]), axis=1)

        return normalized_data
