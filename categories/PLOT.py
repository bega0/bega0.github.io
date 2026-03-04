# 如果是这种格式，使用这个快速可视化
def quick_plot_if_standard_format():
    try:
        # 尝试加载标准格式
        transformer_data = torch.load('Transformer_checkpoint_epoch_30.pth', map_location='cpu')
        lstm_data = torch.load('LSTM_checkpoint_epoch_30.pth', map_location='cpu')
        
        # 提取训练历史
        transformer_train = transformer_data.get('train_losses', [])
        transformer_val = transformer_data.get('val_losses', [])
        lstm_train = lstm_data.get('train_losses', [])
        lstm_val = lstm_data.get('val_losses', [])
        
        # 直接绘图
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(transformer_train, label='Transformer训练')
        plt.plot(transformer_val, label='Transformer验证')
        plt.title('Transformer训练历史')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(lstm_train, label='LSTM训练')
        plt.plot(lstm_val, label='LSTM验证') 
        plt.title('LSTM训练历史')
        plt.legend()
        
        plt.show()
        
    except Exception as e:
        print(f"标准格式加载失败: {e}")