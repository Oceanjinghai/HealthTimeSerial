source_idx=0
accuracy_data_fcn={}
for metric in ['boss','dtw_classic','dtw_sakoechiba','dtw_itakura','dtw_multiscale']:
      mode_name=metric+'+kde'
      accuracy_data_fcn[mode_name]={}
      for i in range(int(data_train.shape[2])):
          view_name = 'view' + str(i)
          locals()[view_name+'_train'] = data_train[:,:,i].reshape(data_train.shape[0],data_train.shape[1],1)
          locals()[view_name+'_test'] = data_test[:,:,i].reshape(data_test.shape[0],data_test.shape[1],1)
      for i in range(int(data_train.shape[2])):
              similarity_name = 'similarity' + str(source_idx)
              if i != source_idx:
                  locals()[similarity_name+str(i)] = cal_similarity(locals()['view'+str(source_idx)+'_train'],locals()['view'+str(i)+'_train'],metric)
          # # KDE
      for i in range(int(data_train.shape[2])):
              kde_name = 'kde'+ str(source_idx)
              similarity_name = 'similarity' + str(source_idx)
              if i != source_idx:
                  locals()[kde_name+str(i)] = KernelDensity(kernel='gaussian', bandwidth=7.8).fit(locals()[similarity_name+str(i)].reshape(locals()[similarity_name+str(i)].flatten().shape[0],1))
      # weight 
      weight_all = 0
      for i in range(int(data_train.shape[2])):
              if i != source_idx:
                  kde_name = 'kde'+ str(source_idx) + str(i)
                  weight_name = 'weight' + str(source_idx) + str(i)
                  locals()[weight_name] =  np.mean(locals()[kde_name].sample(10,random_state=0),axis=0)[0]
                  weight_all += locals()[weight_name]
      for i in range(int(data_train.shape[2])):
        view_name = 'view' + str(i)
        locals()[view_name+'_train'] = data_train[:,:,i].reshape(data_train.shape[0],data_train.shape[1],1)
        locals()[view_name+'_test'] = data_test[:,:,i].reshape(data_test.shape[0],data_test.shape[1],1)
      print('*********************%s No transfer learning********************'%mode_name)

      x, y = build_fcn(view0_train.shape[1:], 2)
      model = keras.models.Model(inputs=x, outputs=y)
      adam = Adam(lr=0.005)
      chk = ModelCheckpoint('best_model.pkl', monitor=tf.keras.metrics.Accuracy(), save_best_only=True, mode='max', verbose=1)
      model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
      history=model.fit(locals()['view'+str(source_idx)+'_train'],
                  to_categorical(target_train),
                  epochs=40,
                  batch_size=16,
                  validation_data=(locals()['view'+str(source_idx)+'_test'],to_categorical(target_test)),verbose=0)
      accuracy_data_fcn[mode_name]['No transfer']=history.history['val_accuracy']
      print(history.history['val_accuracy'][-1])
      print('*********************%s Naive transfer learning********************'%mode_name)
      x, y = build_fcn(view0_train.shape[1:], 2)
      model = keras.models.Model(inputs=x, outputs=y)
      adam = Adam(lr=0.005)
      model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
      
      
      # transfer learning 
      for i in range(int(data_train.shape[2])):
              if i != source_idx:
                  view_name = 'view' + str(i)
                  weight_name = 'weight' + str(source_idx) +str(i)
                  model.fit(locals()[view_name+'_train'], to_categorical(target_train), epochs=30,  batch_size=16,validation_data=(locals()[view_name+'_test'],to_categorical(target_test)),verbose=0)
      # on target domain
      history=model.fit(locals()['view'+str(source_idx)+'_train'],
                  to_categorical(target_train),
                  epochs=30,
                  batch_size=16,
                  validation_data=(locals()['view'+str(source_idx)+'_test'],to_categorical(target_test)),verbose=0)
      accuracy_data_fcn[mode_name]['Naive Transfer']=history.history['val_accuracy']
      print(history.history['val_accuracy'][-1])
      print('*********************%s Weighted transfer learning********************'%mode_name)
      x, y = build_fcn(view0_train.shape[1:], 2)
      model = keras.models.Model(inputs=x, outputs=y)
      adam = Adam(lr=0.005)
      model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
      # transfer learning 
      for i in range(int(data_train.shape[2])):
              if i != source_idx:
                  view_name = 'view' + str(i)
                  weight_name = 'weight' + str(source_idx) +str(i)
                  model.fit(locals()[view_name+'_train'], to_categorical(target_train), epochs=int(30*7*locals()[weight_name]/weight_all)+1,    batch_size=16,validation_data=(locals()[view_name+'_test'],to_categorical(target_test)),verbose=0)
      # on target domain
      history=model.fit(locals()['view'+str(source_idx)+'_train'],
                  to_categorical(target_train),
                  epochs=40,
                  batch_size=16,
                  validation_data=(locals()['view'+str(source_idx)+'_test'],to_categorical(target_test)),verbose=0)
      accuracy_data_fcn[mode_name]['Weighted Transfer']=history.history['val_accuracy']
      print(history.history['val_accuracy'][-1])
      print('\n')

###### A similar training loop


source_idx=1
def cal_similarity(view1,view2,metric='boss'):
    similarity_list = []
    for i in range(view1.shape[0]):
        if metric == 'boss':
            similarity_list.append(boss(np.squeeze(view1[i]),np.squeeze(view2[i])))
        elif metric == 'dtw_classic':
            similarity_list.append(dtw_classic(np.squeeze(view1[i]),np.squeeze(view2[i])))
        elif metric == 'dtw_sakoechiba':
            similarity_list.append(dtw_sakoechiba(np.squeeze(view1[i]),np.squeeze(view2[i]),window_size=0.5))
        elif metric == 'dtw_itakura':
            similarity_list.append(dtw_itakura(np.squeeze(view1[i]),np.squeeze(view2[i]), max_slope=1.5))
        elif metric == 'dtw_multiscale':
            similarity_list.append(dtw_multiscale(np.squeeze(view1[i]),np.squeeze(view2[i]), resolution=2) )
        elif metric == 'dtw_fast':
            similarity_list.append(dtw_fast(np.squeeze(view1[i]),np.squeeze(view2[i]),radius = 1))
        else:
            print('other metric not implement yet.')
    return np.array(similarity_list)


accuracy_data_lstm={}
for metric in ['boss','dtw_classic','dtw_sakoechiba','dtw_itakura','dtw_multiscale']:
      mode_name=metric+'+kde'
      accuracy_data_lstm[mode_name]={}
      for i in range(int(data_train.shape[2])):
          view_name = 'view' + str(i)
          locals()[view_name+'_train'] = data_train[:,:,i].reshape(data_train.shape[0],data_train.shape[1],1)
          locals()[view_name+'_test'] = data_test[:,:,i].reshape(data_test.shape[0],data_test.shape[1],1)
      for i in range(int(data_train.shape[2])):
              similarity_name = 'similarity' + str(source_idx)
              if i != source_idx:
                  locals()[similarity_name+str(i)] = cal_similarity(locals()['view'+str(source_idx)+'_train'],locals()['view'+str(i)+'_train'],metric)
          # # KDE
      for i in range(int(data_train.shape[2])):
              kde_name = 'kde'+ str(source_idx)
              similarity_name = 'similarity' + str(source_idx)
              if i != source_idx:
                  locals()[kde_name+str(i)] = KernelDensity(kernel='gaussian', bandwidth=7.8).fit(locals()[similarity_name+str(i)].reshape(locals()[similarity_name+str(i)].flatten().shape[0],1))
      # weight 
      weight_all = 0
      for i in range(int(data_train.shape[2])):
              if i != source_idx:
                  kde_name = 'kde'+ str(source_idx) + str(i)
                  weight_name = 'weight' + str(source_idx) + str(i)
                  locals()[weight_name] =  np.mean(locals()[kde_name].sample(10,random_state=0),axis=0)[0]
                  weight_all += locals()[weight_name]
      for i in range(int(data_train.shape[2])):
        view_name = 'view' + str(i)
        locals()[view_name+'_train'] = data_train[:,:,i].reshape(data_train.shape[0],data_train.shape[1],1)
        locals()[view_name+'_test'] = data_test[:,:,i].reshape(data_test.shape[0],data_test.shape[1],1)
      print('*********************%s No transfer learning********************'%mode_name)

      x, y = build_lstm(view0_train.shape[1:], 2)
      model = keras.models.Model(inputs=x, outputs=y)
      adam = Adam(lr=0.005)
      chk = ModelCheckpoint('best_model.pkl', monitor=tf.keras.metrics.Accuracy(), save_best_only=True, mode='max', verbose=1)
      model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
      history=model.fit(locals()['view'+str(source_idx)+'_train'],
                  to_categorical(target_train),
                  epochs=40,
                  batch_size=16,
                  validation_data=(locals()['view'+str(source_idx)+'_test'],to_categorical(target_test)),verbose=0)
      accuracy_data_lstm[mode_name]['No transfer']=history.history['val_accuracy']
      print(history.history['val_accuracy'][-1])
      print('*********************%s Naive transfer learning********************'%mode_name)
      x, y = build_lstm(view0_train.shape[1:], 2)
      model = keras.models.Model(inputs=x, outputs=y)
      adam = Adam(lr=0.005)
      chk = ModelCheckpoint('best_model.pkl', monitor=tf.keras.metrics.Accuracy(), save_best_only=True, mode='max', verbose=1)
      model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
      
      
      # transfer learning 
      for i in range(int(data_train.shape[2])):
              if i != source_idx:
                  view_name = 'view' + str(i)
                  weight_name = 'weight' + str(source_idx) +str(i)
                  model.fit(locals()[view_name+'_train'], to_categorical(target_train), epochs=30,  batch_size=16,validation_data=(locals()[view_name+'_test'],to_categorical(target_test)),verbose=0)
      # on target domain
      history=model.fit(locals()['view'+str(source_idx)+'_train'],
                  to_categorical(target_train),
                  epochs=30,
                  batch_size=16,
                  validation_data=(locals()['view'+str(source_idx)+'_test'],to_categorical(target_test)),verbose=0)
      accuracy_data_lstm[mode_name]['Naive Transfer']=history.history['val_accuracy']
      print(history.history['val_accuracy'][-1])
      print('*********************%s Weighted transfer learning********************'%mode_name)
      x, y = build_lstm(view0_train.shape[1:], 2)
      model = keras.models.Model(inputs=x, outputs=y)
      adam = Adam(lr=0.005)
      chk = ModelCheckpoint('best_model.pkl', monitor=tf.keras.metrics.Accuracy(), save_best_only=True, mode='max', verbose=0)
      model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
      # transfer learning 
      for i in range(int(data_train.shape[2])):
              if i != source_idx:
                  view_name = 'view' + str(i)
                  weight_name = 'weight' + str(source_idx) +str(i)
                  model.fit(locals()[view_name+'_train'], to_categorical(target_train), epochs=int(30*7*locals()[weight_name]/weight_all)+1,    batch_size=16,validation_data=(locals()[view_name+'_test'],to_categorical(target_test)),verbose=0)
      # on target domain
      history=model.fit(locals()['view'+str(source_idx)+'_train'],
                  to_categorical(target_train),
                  epochs=40,
                  batch_size=16,
                  validation_data=(locals()['view'+str(source_idx)+'_test'],to_categorical(target_test)),verbose=0)
      accuracy_data_lstm[mode_name]['Weighted Transfer']=history.history['val_accuracy']
      print(history.history['val_accuracy'][-1])
      print('\n')
