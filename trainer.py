from tqdm import tqdm
from torch.nn import functional as F
from utils import utils as u


def train_one_image(model_x, model_k, epochs, optimizer, criterion,
                    train_metrics, target, device, data_x, data_k):
    # training-the-model
    train_loss = 0
    model_x.train()
    model_k.train()
    print('*****************************')
    for epoch in tqdm(range(epochs)):
        print("starting epoch ", i)
        # move-tensors-to-GPU
        data = data.to(device)
        # target=torch.Tensor(target)
        target = target.to(device)
        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output_x = model_x(data_x)
        # get the prediction label and target label
        output_k = model_k(data_k)
        # re-constructed deblurred image
        deblured = F.conv2d(output_x, output_k, padding=0, bias=None)
        # calculate-the-batch-loss
        loss = criterion(deblured, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item()
        print('current_loss : ', loss.item())
        # calculate training metrics
        train_metrics.step(target, deblured)

    return (
        train_loss,
        train_metrics.epoch(),
    )


def train_one_epoch(
    model,
    train_loader,
    device,
    optimizer,
    criterion,
    train_metrics,
    val_metrics,
):

    # training-the-model
    train_loss = 0
    valid_loss = 0
    model.train()
    for data, target in train_loader:
        # move-tensors-to-GPU
        data = data.to(device)
        # target=torch.Tensor(target)
        target = target.to(device)
        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # get the prediction label and target label
        output = model(data)
        preds = torch.argmax(output, axis=1).cpu().detach().numpy()
        labels = target.cpu().numpy()
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)
        # calculate training metrics
        train_metrics.step(labels, preds)

    return (
        train_loss,
        valid_loss,
        train_metrics.epoch(),
        val_metrics.last_step_metrics(),
    )
