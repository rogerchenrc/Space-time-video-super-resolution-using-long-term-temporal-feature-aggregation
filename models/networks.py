import models.modules.NeXtSTVSR as STVSR

####################
# define network
####################
# Generator
def define_G(opt, device):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'NextMoENet':
        # need to change to my version
        netG = STVSR.NextMoENet(nf=opt_net['nf'], nframes=opt_net['nframes'], groups=opt_net['groups'], front_RBs=opt_net['front_RBs'], back_RBs=opt_net['back_RBs'], opt=opt, device=device)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
