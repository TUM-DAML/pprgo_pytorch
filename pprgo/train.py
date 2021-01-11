import logging
import numpy as np
import torch
import torch.nn.functional as F


def run_batch(model, xbs, yb, optimizer, train):

    # Set model to training mode
    if train:
        model.train()
    else:
        model.eval()

    # zero the parameter gradients
    if train:
        optimizer.zero_grad()

    # forward
    with torch.set_grad_enabled(train):
        pred = model(*xbs)
        loss = F.cross_entropy(pred, yb)
        top1 = torch.argmax(pred, dim=1)
        ncorrect = torch.sum(top1 == yb)

        # backward + optimize only if in training phase
        if train:
            loss.backward()
            optimizer.step()

    return loss, ncorrect


def train(model, train_set, val_set, lr, weight_decay,
          max_epochs=200, batch_size=512, batch_mult_val=4,
          eval_step=1, early_stop=False, patience=50, ex=None):
    device = next(model.parameters()).device

    train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(train_set),
                batch_size=batch_size, drop_last=False
            ),
            batch_size=None,
            num_workers=0,
    )
    step = 0
    best_loss = np.inf

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}
    if ex is not None:
        ex.current_run.info['train'] = {'loss': [], 'acc': []}
        ex.current_run.info['val'] = {'loss': [], 'acc': []}

    loss = 0
    ncorrect = 0
    nsamples = 0
    for epoch in range(max_epochs):
        for xbs, yb in train_loader:
            xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)

            loss_batch, ncorr_batch = run_batch(model, xbs, yb, optimizer, train=True)
            loss += loss_batch
            ncorrect += ncorr_batch
            nsamples += yb.shape[0]

            step += 1
            if step % eval_step == 0:
                # update train stats
                train_loss = loss / nsamples
                train_acc = ncorrect / nsamples

                loss_hist['train'].append(train_loss)
                acc_hist['train'].append(train_acc)
                if ex is not None:
                    ex.current_run.info['train']['loss'].append(train_loss)
                    ex.current_run.info['train']['acc'].append(train_acc)

                if val_set is not None:
                    # update val stats
                    rnd_idx = np.random.choice(len(val_set), size=batch_mult_val * batch_size, replace=False)
                    xbs, yb = val_set[rnd_idx]
                    xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)
                    val_loss, val_ncorr = run_batch(model, xbs, yb, None, train=False)
                    val_acc = val_ncorr / (batch_mult_val * batch_size)

                    loss_hist['val'].append(val_loss)
                    acc_hist['val'].append(val_acc)
                    if ex is not None:
                        ex.current_run.info['val']['loss'].append(val_loss)
                        ex.current_run.info['val']['acc'].append(val_acc)

                    logging.info(f"Epoch {epoch}, step {step}: train {train_loss:.5f}, val {val_loss:.5f}")

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_epoch = epoch
                        best_state = {
                                key: value.cpu() for key, value
                                in model.state_dict().items()
                        }
                    # early stop only if this variable is set to True
                    elif early_stop and epoch >= best_epoch + patience:
                        model.load_state_dict(best_state)
                        return epoch + 1, loss_hist, acc_hist
                else:
                    logging.info(f"Epoch {epoch}, step {step}: train {train_loss:.5f}")
    if val_set is not None:
        model.load_state_dict(best_state)
    return epoch + 1, loss_hist, acc_hist
