def lstm_training(train_data_normalized, device, scaler, timestep, patience, \
                  epochs, B, input_dim, hidden_dim, output_dim, num_layers):
    
    p_train = pd.DataFrame()    
    r = []

    # restructure the training and test datasets to sequeces 
    train_seq_label = create_inout_sequences(train_data_normalized, timestep)
    train_inout_seq, train_label = split_inout_sequences(train_seq_label,timestep)
    train_inout_seq_input = Variable(torch.Tensor(np.array(train_inout_seq)))
    train_label_inverse = scaler.inverse_transform(train_label)

    for b in range(B):
        last_loss = 100
        trigger_times = 0

        # resample the restructured training dataset and get the OBB set for validating
        train_seq_label_b = resample(train_seq_label,n_samples=int(len(train_seq_label)))
        test_seq_label_b = NotInB(train_seq_label,train_seq_label_b)

        train_inout_seq_b, train_label_b  = split_inout_sequences(train_seq_label_b,timestep)
        test_inout_seq_b, test_label_b = split_inout_sequences(test_seq_label_b,timestep)
        train_inout_seq_b = Variable(torch.Tensor(np.array(train_inout_seq_b)))

        train_label_b = Variable(torch.Tensor(np.array(train_label_b)))
        test_inout_seq_b = Variable(torch.Tensor(np.array(test_inout_seq_b)))
        test_label_b = Variable(torch.Tensor(np.array(test_label_b)))

        # train the LSTM model
        model = LSTM(input_dim=input_dim,hidden_dim=hidden_dim,
                    output_dim=output_dim,num_layers=num_layers)
        model.to(device)
        loss_fn = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for i in range(epochs): 
            model.train()
            optimizer.zero_grad()
            y_pred = model(train_inout_seq_b.to(device))
            single_loss = loss_fn(y_pred, train_label_b.to(device))
            single_loss.backward()
            optimizer.step()
            model.eval()
            vaild = model(test_inout_seq_b.to(device))
            valid_loss = loss_fn(vaild, test_label_b.to(device))

            # using earlystop
            current_loss = valid_loss.item()
            if current_loss > last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    break
            else:
                trigger_times = 0
            last_loss = current_loss

        # get the prediction on training set
        y_train_pred = model(train_inout_seq_input.to(device))
        train_pred = scaler.inverse_transform(y_train_pred.cpu().data.numpy())
        p_train = pd.concat([pd.DataFrame(train_pred),p_train],axis=1)
    # calculate the mean of predictions
    p_train_mean = np.reshape(np.array(p_train.mean(axis=1)),(-1,1))

    # estimate the sigma and r for the STD-ANN

    sig = np.zeros([len(p_train_mean),1], dtype=float) 
    for _,col in p_train.items():
        col_i=np.array(col).reshape(-1,1)
        sig = np.sum([sig,np.square(col_i-p_train_mean)],axis=0)

    sig = np.array(sig)/(B-1)
    t_r = np.square(train_label_inverse-p_train_mean)-sig
    for x in t_r:
        x = max(x,np.zeros([1,], dtype=int))
        r.append(x)
    r = np.array(r,dtype=float)

    return p_train_mean, sig, r


def ann_pi_building(train_data_normalized, p_train_mean, r, sig, scaler,\
                    timestep, z):
    # restructure the data for ANN
    train_seq_label_ann = create_inout_sequences(scaler.inverse_transform(train_data_normalized), timestep)
    train_inout_seq_ann, _ = split_inout_sequences_ann(train_seq_label_ann,timestep)
    train_inout_seq_ann_input = Variable(torch.Tensor(train_inout_seq_ann))
    r_input  = Variable(torch.Tensor(r))


    # define the ANN and optimizer
    model_B = sigma(input_size_1=timestep)
    optimizer_B = torch.optim.Adam(model_B.parameters(), lr=1e-2)
    epochs=500
    # train the ANN
    for i in range(epochs): 
        optimizer_B.zero_grad()
        sig_ = model_B(train_inout_seq_ann_input)
        loss_B = cbs(sig_, r_input)
        loss_B.backward()
        optimizer_B.step()

    # combine the two parts of variance
    pi_train = np.sqrt(np.sum([np.array(model_B(train_inout_seq_ann_input).tolist()),sig],axis=0))
    u_pi_train = p_train_mean + z*pi_train
    l_pi_train = p_train_mean - z*pi_train

    return u_pi_train,l_pi_train