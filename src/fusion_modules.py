import torch



class DummyLayer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x1, x2, dem=None):
        return torch.empty_like(x1)
    

    
class SSMA(torch.nn.Module):

    def __init__(self, channels, eta=4, **kwargs):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2*channels, channels//eta, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(channels//eta, 2*channels, kernel_size=3, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv3 = torch.nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        # get SSMA weights from x
        ssma_weights = self.conv1(x)
        ssma_weights = self.relu(ssma_weights)
        ssma_weights = self.conv2(ssma_weights)
        ssma_weights = self.sigmoid(ssma_weights)

        # apply SSMA weights to x
        x = x * ssma_weights

        x = self.conv3(x)
        x = self.bn(x)
        
        return x



class DiffFeatures(torch.nn.Module):
    def __init__(self, out_channels=None, **kwargs):
        super().__init__()
        if out_channels is not None:
            conv = torch.nn.Conv2d(out_channels+2, out_channels, kernel_size=3, padding=1, bias=True)
            bn = torch.nn.BatchNorm2d(out_channels)
            relu = torch.nn.ReLU()
            self.conv_module = torch.nn.Sequential(conv, bn, relu)

    def forward(self, x1, x2, dem=None):
        diffs = x2 - x1
        if dem is None:
            return diffs
        else:
            f = torch.cat([diffs, dem], dim=1)
            return self.conv_module(f)
        


class AbsDiffFeatures(torch.nn.Module):
    def __init__(self, out_channels=None, **kwargs):
        super().__init__()
        if out_channels is not None:
            conv = torch.nn.Conv2d(out_channels+2, out_channels, kernel_size=3, padding=1, bias=True)
            bn = torch.nn.BatchNorm2d(out_channels)
            relu = torch.nn.ReLU()
            self.conv_module = torch.nn.Sequential(conv, bn, relu)

    def forward(self, x1, x2, dem=None):
        diffs = torch.abs(x2 - x1)
        if dem is None:
            return diffs
        else:
            f = torch.cat([diffs, dem], dim=1)
            return self.conv_module(f)



class ChangeformerDiffFeatures(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.diff_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # present in Changeformer implementation but not in paper
            torch.nn.ReLU()                                                         # present in Changeformer implementation but not in paper
    )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.diff_module(x)
        return x



class ConcatFeatures(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x1, x2):
        concats = torch.cat([x1, x2], dim=1)
        return concats