import torch
from loss.loss import loss_vb


def train(model, dataloader, optimizer, run, exp_id, epochs, sigma=1e-2, scaling_term=2e4, epsilon=1e-2):
    model.train()
    size = len(dataloader.dataset)
    total_loss = torch.tensor(0.0, device="cuda")
    for batch, (blur_kernel, blur_img, latent_img) in enumerate(dataloader):
        blur_kernel, blur_img, latent_img = blur_kernel.to("cuda"), blur_img.to("cuda"), latent_img.to("cuda")
        mu, log_var, simplex, blur_kernel_gen, latent_img_gen = model(blur_img)
        loss = loss_vb(blur_kernel,
                       blur_img,
                       latent_img,
                       mu,
                       log_var,
                       simplex,
                       blur_kernel_gen,
                       latent_img_gen,
                       sigma,
                       scaling_term,
                       epsilon)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(blur_kernel)
            print("loss: {}  [{}/{}]".format(loss, current, size))
    ave_loss = total_loss / len(dataloader)
    # log information of loss
    run["train/loss"].log(ave_loss)
    print("average loss: {}".format(ave_loss))
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, '/notebooks/checkpoints/{}.pth'.format(exp_id))
    run["model"].upload("/notebooks/checkpoints/{}.pth".format(exp_id)) #models
    return ave_loss
