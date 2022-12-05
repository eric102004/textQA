


def warmup_linear(step, total, warmup_step=100, ends=0):
    x = step/total
    x = x-int(x)
    if x < warmup_step/total:
        return x/warmup_step*total
    return (1-ends)*(1.0 - x) + ends

def update_lr(optimizer, max_lr, step, total, warmup_step=100, ends=0, schedule_fn=warmup_linear):
    lr_schedule = schedule_fn(step, total, warmup_step=warmup_step, ends=ends)
    new_lr = max_lr * lr_schedule
    # update learning rate
    optimizer.param_groups[0]['lr'] = new_lr
    
