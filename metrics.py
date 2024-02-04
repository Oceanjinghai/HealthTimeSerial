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
