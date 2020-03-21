'''
This module serves for the visualization
'''
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_latent_map(encode_latent , \
                        latent_label = None, \
                        output_dir   = None, \
                        step  = '0', \
                        title = 'Latent map', name='lmap', plotid=111):
    if output_dir == None:
        print('>> [generate_latent_map] Output folder is required\
                                            save visualization figure.')
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    '''
    Plot latent points
    ''' 
    mbsize, dim = np.shape(encode_latent)
    if dim > 2 or dim < 2:
        print('>> [generate_latent_map] Latent dimension mus be 2!!')
        return
        
    plt.figure(figsize=(10,10))
    kwargs = {'alpha': 0.8}
    
    if latent_label is not None:
        latent_label = latent_label.astype(int).flatten()
        classes = set(latent_label)
        colormap = plt.cm.rainbow(np.linspace(0, 1, 10))
        kwargs['c'] = [colormap[i] for i in latent_label]
        
    ax  = plt.subplot(plotid)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    
    if latent_label is not None:
        handles = [mpatches.Circle((0,0), label=class_, color=colormap[i]) for i, class_ in enumerate(classes)]
        ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45), fancybox=True, loc='center left')

    plt.scatter(encode_latent[:,0], encode_latent[:,1], **kwargs)
    plt.title(title)
    
    plt.savefig(output_dir + "/" + name +  "_step_%d.jpg" % (step),  \
                                                    bbox_inches='tight')
    plt.close()
    
                                   
 
                              
                              
                              


