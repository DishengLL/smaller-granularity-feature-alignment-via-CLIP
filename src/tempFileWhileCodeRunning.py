class Pearson_correlation_loss(nn.Module):
    '''
    Pearson correlation loss is aiming to measure the correlation between two tensor x and y with shape of [N-diseases, N-diseases]
    
    def forward(self, x, y): input tensor x and tensor y and return the final loss between these two tensor
    
    the value range of Pearson correlation is [-1, 1], and in the loss function, I use loss = 1 - cost, so the range of loss is [0, 2]
    '''
    def __init__(self):
        super(Pearson_correlation_loss, self).__init__()

    def forward(self, x, y):
      """
      x and y are tensors with shape of [N-diseases, N-diseases]
      
      """
        x_mean = torch.mean(output, dim=1, keepdim=True)
        y_mean = torch.mean(target, dim=1, keepdim=True)
        output_centered = x - output_mean
        target_centered = y - target_mean
        
        numerator = torch.sum(output_centered * target_centered, dim=1)
        denominator = torch.sqrt(torch.sum(output_centered ** 2, dim=1)) * torch.sqrt(torch.sum(target_centered ** 2, dim=1))
        
        correlation = numerator / (denominator + 1e-8)  # 防止除以0
        loss = 1 - correlation  # 使得最大化相关性最小化损失
        
        return loss.mean()
        # cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        # return 1 - cost
        
        
        
# 在当前的实验中，加上contrastive loss之后，模型的效果并没有明显提升，所以我决定尝试一下其他的loss function，这里我选择了Pearson correlation loss，这个loss function是用来衡量两个tensor之间的相关性的，我认为这个loss function可能会对模型的效果有所提升，所以我决定尝试一下。

'''
loss
1. classification loss: cross entropy loss
2. contrastive loss: contrastive loss
3. orthogonality loss: orthogonality loss
4. high-order correlation: Pearson correlation loss

ablition study:
1. classification loss

2. classification loss + contrastive loss
3. classification loss + Pearson correlation loss


4. classification loss + contrastive loss + orthogonality loss
5. classification loss + contrastive loss + Pearson correlation loss

4. classification loss + contrastive loss + orthogonality loss + Pearson correlation loss


the experiment of hyperparameter tuning:
whether fix the hyperperameter of aulixiary loss in a small value obtained a better result than tuning the hyperparameter of aulixiary loss
'''