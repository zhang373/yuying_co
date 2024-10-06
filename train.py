from dataset.utils import prepare_data
from model import train
import argparse
path_dict = {"2": "AI-Usage",
                "3": "Humanlikeness_Mental",
                "4": "Humanlikeness_Visual",
                "5": "PSI_Object_AI",
                "6": "PSI_object_charactor_itself",
                "7": "PSI_Agreement",
                "8": "PSI_Eexpress_opinion",
                "9":"PSI_group",
                "10": "PSI_intersted",
                "11": "AI_Merged"}

if __name__ == "__main__":
    """
    data selection from raw data file, you should choose index from [2: 12] and modify the input data by modifying the label_columns.
    """
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument("--label_columns", type=int, default=2, help="train on which cloumn, from [2, 11], [] is math samble")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--eval_rounds", type=int, default=5, help="Evaluation frequency.")
    parser.add_argument("--assign_gate", type=float, default=0.3, help="Threshold for assigning predictions.")
    parser.add_argument("--device", type=str, default='cuda:5', help="Device to use for training.")
    parser.add_argument("--gamma", type=float, default='1', help="How much importance on hard data.")
    parser.add_argument("--alpha", type=float, default='0.75', help="How much to balance on 1 labeled data.")
    args = parser.parse_args()

    
    label_columns = args.label_columns

    savingpath = "./results/"+path_dict[str(label_columns)]+"/"

    print(f"We have entered the train.py file and do training now, we are working on {path_dict[str(label_columns)]}, we will process the data")

    train_dataset, test_dataset, model = prepare_data(label_columns=label_columns, device=args.device)
    print(f"We have finish model preparation and will start training model")

    train(train_dataset, test_dataset, model, batch_size=args.batch_size, num_epochs=args.num_epochs, lr=args.lr,
          eval_rounds=args.eval_rounds, assign_gate=args.assign_gate, savingpath=savingpath, device=args.device, gamma=args.gamma, alpha=args.alpha)