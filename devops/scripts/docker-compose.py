import os
import argparse
from pathlib import Path
import subprocess


SCRIPTPATH = Path(__file__).parent.resolve()
def build(profile: str):
    subprocess.run(f'sudo docker compose -f docker-compose.yaml --env-file .env.{profile} build', shell=True)

def up(profile: str, service: str | None):
    service_str = "" if service is None else service
    subprocess.run(f'sudo docker compose -f docker-compose.yaml --env-file .env.{profile} up {service_str} --build --remove-orphans --force-recreate --wait', shell=True)

def down(profile: str, service: str | None):
    service_str = "" if service is None else service
    subprocess.run(f'sudo docker compose -f docker-compose.yaml --env-file .env.{profile} down {service_str} --remove-orphans --volumes', shell=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=["up", "down", "build"])
    parser.add_argument('profile', type=str, choices=["local", "dev"])
    parser.add_argument('-s', '--service', type=str, dest='service', required=False, choices=["thapo-ska-gateway", "thapo-ska-backend", "thapo-ska-frontend"])
    args = parser.parse_args()

    os.chdir(SCRIPTPATH.joinpath("../docker").resolve())
    if args.action == "down":
        down(args.profile, args.service)
    elif args.action == "up":
        up(args.profile, args.service)
    elif args.action == "build":
        build(args.profile)




if __name__ == "__main__":
    main()