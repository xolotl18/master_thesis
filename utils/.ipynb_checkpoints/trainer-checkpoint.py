#maybe try and make it a class to that you can output the model and perform training and validation internally
#if the epoch iteration is performed in this methods, the selection of the best model can be done internally




def train(train_loader, model, criterion, optimizer, scheduler, epoch, params):
  model.train()
  running_loss = 0.0
  stream = tqdm(train_loader)
  for i, data in enumerate(stream, 1):
        optimizer.zero_grad()
        
        images, targets = data
        images = images.to(params["device"], non_blocking=True)
        targets = targets.to(params["device"], non_blocking=True)

        outputs = model(images)
        targets = torch.unsqueeze(targets, 1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        running_loss += loss.item()*images.size(0)
        stream.set_description(
            "Epoch: {epoch}. Train.    {loss:.5f}".format(epoch=epoch, loss=running_loss)
        )

def validate(val_loader, model, criterion, epoch, params):
  model.eval()
  running_loss=0.0
  stream = tqdm(val_loader)
  with torch.no_grad():
    for i, data in enumerate(stream, start=1):
            images, targets = data
            images = images.to(params["device"], non_blocking=True)
            targets = targets.to(params["device"], non_blocking=True)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
            running_loss += loss.item()*images.size(0)
            stream.set_description(
                "Epoch: {epoch}. Validation. {loss:.5f}".format(epoch=epoch, loss=running_loss)
            )
  return running_loss

def train_deeplab(train_loader, model, criterion, optimizer, scheduler, epoch, params):
  model.train()
  running_loss = 0.0
  stream = tqdm(train_loader)
  for i, data in enumerate(stream, 1):
        optimizer.zero_grad()
        
        images, targets = data
        images = images.to(params["device"], non_blocking=True)
        targets = targets.to(params["device"], non_blocking=True)

        outputs = model(images)
        targets = torch.unsqueeze(targets, 1)
        
        loss = criterion(outputs['out'], targets)
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        running_loss += loss.item()*images.size(0)
        stream.set_description(
            "Epoch: {epoch}. Train.    {loss:.5f}".format(epoch=epoch, loss=running_loss)
        )

def validate_deeplab(val_loader, model, criterion, epoch, params):
  model.eval()
  running_loss=0.0
  stream = tqdm(val_loader)
  with torch.no_grad():
    for i, data in enumerate(stream, start=1):
            images, targets = data
            images = images.to(params["device"], non_blocking=True)
            targets = targets.to(params["device"], non_blocking=True)
            targets = torch.unsqueeze(targets, 1)
            outputs = model(images)
            loss = criterion(outputs['out'], targets)
            running_loss += loss.item()*images.size(0)
            stream.set_description(
                "Epoch: {epoch}. Validation. {loss:.5f}".format(epoch=epoch, loss=running_loss)
            )
  return running_loss 