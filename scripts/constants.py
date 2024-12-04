import torch

csv_train_ans = 'C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\human_poses_data\\train_answers.csv'
root_dir_train = 'C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\human_poses_data\\img_train'
root_dir_test = 'C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\human_poses_data\\img_test'
hist_path = "hist"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")