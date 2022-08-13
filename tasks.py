from invoke import task

@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")

@task(aliases=['del'])
def delete(c):
    c.run("rm mykmeanssp.so")

@task
def run(c, n = "-1", k = "-1", Random = True):
    build(c)
    c.run("python3.8.5 main.py {n:s} {k:s} {random:}".format(n=n, k=k, random=Random))