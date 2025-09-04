def test(model, data, device):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask].to(device)).sum()
    accuracy = int(correct) / int(data.test_mask.sum())
    return accuracy