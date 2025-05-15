from model import GPTLanguageModel
import grain.python as grain

from src.Notebooks.Transformers.Shakspeare.data import eval_interval
from src.Transformer.Data import Split
import torch



eval_interval = 500
eval_iters = 200

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else :
    device = torch.device('cpu')

learning_rate = 3e-4
eval_interval = 500
batch_size = 64

train_data = Split('train', 'long',batch_size)

val_data = Split('val', 'long',batch_size)

model = GPTLanguageModel()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
@torch.no_grad()
def estimate_loss(element):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = element #TODO see what the shape of each batch actually is
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    for element, step in enumerate(train_data.data_loader):
        if step % 100 == 0:
            losses = estimate_loss()
            print(f"step {step} train loss: {losses}, val loss: {losses}")

        xb, yb = element #given 256 predict the next 32 so do something here idk

        logits, loss = model(xb, yb) #TODO change this
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'model.pt')
    print("model saved")
