# `commands`

CLI handlers live here.

## Contents

- command wiring for `lcm`
- subcommand-specific argument parsing and dispatch
- small provider utilities shared by command handlers

## Maintenance Rule

Keep this layer thin. It should translate CLI arguments into calls to the
underlying packages, not reimplement workflow logic.
