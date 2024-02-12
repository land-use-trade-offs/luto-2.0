from retrying import retry

@retry
def run():
    print("Begin")
    import luto.simulation as sim
    import luto.tools.write as write
    import luto.settings as settings
    nums = [0]
    MODEs = ['timeseries','timeseries','timeseries','snapshot','snapshot','snapshot']
    RESFACTORs = [30,3,3,1,1,1]
    TARGETs = [-51,-129,-337.5,-51,-129,-337.5]

    for i in nums:
        print(f"MODE = {MODEs[i]},RESFACTOR = {RESFACTORs[i]},TARGET = {TARGETs[i]}")

        settings.MODE = MODEs[i] # 'snapshot''timeseries'
        settings.RESFACTOR = RESFACTORs[i]
        settings.GHG_LIMITS[2050] = TARGETs[i] * 1e6

        sim.run(2010, 2050)
        path = write.get_path(sim)
        if MODEs[i] == 'snapshot':
            write.write_outputs_snapshot(sim, path)
        else:
            write.write_outputs(sim, path)

print(1)
run()

