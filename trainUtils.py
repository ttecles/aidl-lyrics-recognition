def accuracy(predicted_batch, ground_truth_batch):
  pred = predicted_batch.argmax(dim=1, keepdim=True) # get the index of the max log-probability
  acum = pred.eq(ground_truth_batch.view_as(pred)).sum().item()
  return acum