from sample import conversation_sampler

if __name__ == '__main__':
    sampler = conversation_sampler()
    conv_it = sampler.get_next()
    inp = ''
    while inp != 'exit':
        print(next(conv_it))
        inp = input('Type "exit" to stop: ')
        