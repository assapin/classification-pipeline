

class BasicEvaluator:
    def __init__(self, model, val_loader):
        self.model = model
        self.val_loader = val_loader

    def validate(self):
        score = self.model.evaluate(self.val_loader, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])