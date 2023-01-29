import os
import os.path
import csv


class FileMetaData:
    def __init__(self, base_path, train_folder, test_folder, target_folder):
        self.base_path = base_path
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.target_folder = target_folder
        self.train_csv_file_name = 'train_csv.csv'
        self.test_csv_file_name = 'test_csv.csv'

        if os.path.isfile(self.base_path + "\\" + self.train_csv_file_name):
            os.remove(self.base_path + "\\" + self.train_csv_file_name)
        if os.path.isfile(self.base_path + "\\" + self.test_csv_file_name):
            os.remove(self.base_path + "\\" + self.test_csv_file_name)

    def process_metadata(self):
        test_folder_base = self.base_path + self.test_folder
        test_data_dir = [data_dir[0] for data_dir in os.walk(test_folder_base)][1:]

        with open(self.base_path + "\\" + self.test_csv_file_name, 'w') as test_csv_file_handle:
            test_csv_file_writer = csv.writer(test_csv_file_handle)

            for data_dir in test_data_dir:
                file_label = data_dir.split("\\")[-1]
                for file in os.listdir(data_dir):
                    test_csv_row = [file, file_label]
                    test_csv_file_writer.writerow(test_csv_row)

        train_folder_base = self.base_path + self.train_folder
        train_data_dir = [data_dir[0] for data_dir in os.walk(train_folder_base)][1:]

        with open(self.base_path + "\\" + self.train_csv_file_name, 'w') as train_csv_file_handle:
            train_csv_file_writer = csv.writer(train_csv_file_handle)

            for data_dir in train_data_dir:
                file_label = data_dir.split("\\")[-1]
                for file in os.listdir(data_dir):
                    train_csv_row = [file, file_label]
                    train_csv_file_writer.writerow(train_csv_row)