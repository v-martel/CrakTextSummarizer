class Digestible(object):
    def __init__(self, inputs: [str], outputs: [str]):
        self.inputs = inputs
        self.outputs = outputs

    def __str__(self):
        return 'inputs are {} \nouputs are {}'.format(self.inputs, self.outputs)
