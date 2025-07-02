from data.order_dataset import OrderFreqDataset
from models.segment_transformer import SegmentLevelModel
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from tqdm import tqdm


if __name__ == '__main__':
    batch_size = 32
    num_epoch = 10
    device = 'cuda'
    
    dataset = OrderFreqDataset(
        data_root= '/data/vms_dataset', 
        classes = ['normal', 'looseness', 'misalignment', 'unbalance', 'bearing'], 
        averaging_size = 100, 
        target_len=260, 
        sensor_list=['motor_x', 'motor_y'], 
        max_order = 10
    )
    
    model = SegmentLevelModel(
        embed_dim=64,
        n_heads=4,
        num_segments=10,
        num_classes=5
    ).to(device)
    
    data_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True)

    optim = Adam(model.parameters())

    for epoch in range(num_epoch):
        epoch_loss = 0.0
        for batch in tqdm(data_loader):
            sample_tensor, normal_tensor, class_tensor = batch.to(device)
            
            pred, attn = model(sample_tensor, normal_tensor)
            
            loss = cross_entropy(pred, class_tensor)
            
            loss.backward()
            optim.step()
            
            epoch_loss += loss.sum().item()
        print(f'epoch : {epoch} // loss : {epoch_loss}')