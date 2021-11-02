import argparse
import neptune.new as neptune
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from dataloader.TextDataset import Text
from model.VBDeblur import VB_Blur
from train.train import train
from predict.resume import resume, test
from torchsummary import summary


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', type=str, default='exp_8')

    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    
    # image size
    parser.add_argument('--img_size', type=int, default=256, help='image size')

    # train
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1e0, help="learning rate decay after each epochs")

    # loss
    parser.add_argument('--simplex_scale', type=float, default=2e4, help="scale the simplex of blur kernel distribution")
    parser.add_argument('--sigma', type=float, default=1e-2, help="sigma")
    parser.add_argument('--epsilon', type=float, default=1e-2, help="epsilon")

    # model
    parser.add_argument('--res_number', type=int, default=2, help="number of resblock in encoder/decoder block")
    parser.add_argument('--channel_number', type=int, default=64, help="number of channels in encoder/decoder block")
    parser.add_argument('--blur_channel_number', type=int, default=10, help="number of blur channels sft layer in "
                                                                           "encoder/decoder block")
    # seed
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    img_transform = Compose([
        Resize((args.img_size, args.img_size)),
        ToTensor(),
    ])

    blur_transform = Compose([
        Resize((17, 17)),
        ToTensor(),
    ])

    text = Text("/notebooks/dataset/text/data/", img_transform, blur_transform)
    no_train = int(0.8 * len(text))
    no_test = len(text) - no_train
    train_data, test_data = torch.utils.data.random_split(text, [no_train, no_test])
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=2)

    torch.cuda.empty_cache()
    model = VB_Blur(img_size=args.img_size,
                    kernel_size=17,
                    n_res_block=args.res_number,
                    channel=args.channel_number,
                    blur_channel=args.blur_channel_number)
    
    

    model = nn.DataParallel(model).to("cuda")
    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay, verbose=True)
    
    if args.resume:
        model, optimizer = resume(args.exp_id, model, optimizer)
        
    if args.test:
        test(model, text, args.img_size)
    else:
        run = neptune.init(project="kevinqd/image-deblur", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZDYzNWE1NS05NjliLTQ5YjQtYmRhNS0xNTE2NzNlN2E2NjEifQ==",)
        params = {"learning_rate": args.lr, 
              "exp_id": args.exp_id, 
              "batch_size": args.batch_size, 
              "epochs": args.epochs,
              "lr_decay": args.lr_decay,
              "simplex_scale": args.simplex_scale,
              "sigma": args.sigma,
              "epsilon": args.epsilon,
              "res_number": args.res_number,
              "channel_number": args.channel_number,
              "blur_channel_number": args.blur_channel_number
             }
        run["parameters"] = params
        for t in range(args.epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            hist = train(model,
                         test_dataloader,
                         optimizer,
                         run,
                         exp_id=args.exp_id,
                         epochs=t,
                         sigma=args.sigma,
                         scaling_term=args.simplex_scale,
                         epsilon=args.epsilon)
            scheduler.step()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
