class EarlyStopping:
    def __init__(self,patience=10,verbose=0):
        """_summary_

        Args:
            patience (int): 監視するエポック数(デフォルト 10)
            verbose (int): 早期終了の出力フラグ 1:出力 2:出力しない
        """
        self.epoch = 0
        self.pre_loss = float('inf')
        self.patience = patience
        self.verbose = verbose
    
    def __call__(self,current_loss):
        """_summary_

        Args:
            current_loss (float): 1エポック終了後の検証データのloss
        
        Return: 
            True:監視回数の上限までに前エポックの損失を超えた場合
            False:監視回数の上限までに前エポックの損失を超えない場合
        """
        if self.pre_loss < current_loss: #前エポックよりもlossが小さい場合
            self.epoch += 1

            if self.epoch > self.patience:
                if self.verbose:
                    print("early stopping")
                return True
        else: #前エポックよりもlossが小さい場合
            self.epoch = 0
            self.pre_loss = current_loss
        
        return False



