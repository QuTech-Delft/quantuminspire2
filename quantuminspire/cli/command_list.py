"""Module containing the commands for the Quantum Inspire 2 CLI."""
from enum import Enum
from typing import Optional

import typer
from typer import Typer

app = Typer(add_completion=False)
algorithms_app = Typer()
app.add_typer(algorithms_app, name="algorithms", help="Manage algorithms")
configuration_app = Typer()
app.add_typer(configuration_app, name="config", help="Manage configuration")
projects_app = Typer()
app.add_typer(projects_app, name="projects", help="Manage projects")


class Destination(str, Enum):
    """Enumeration of potential destinations for projects."""

    LOCAL = "local"
    REMOTE = "remote"


@algorithms_app.command("create")
def create_algorithm(name: str, hybrid: bool = typer.Option(False)) -> None:
    """Create a new algorithm.

    Create an algorithm with a name. Depending on the hybrid flag,
    either the "hybrid algorithm" or the "quantum circuit" template is
    used.
    """
    if hybrid:
        typer.echo(f"Creating hybrid quantum/classical algorithm '{name}'")
    else:
        typer.echo(f"Creating quantum circuit '{name}'")


@algorithms_app.command("commit")
def commit_algorithm() -> None:
    """Commit algorithm to API.

    The algorithm is uploaded to the remote API. The algorithm is
    selected on the folder, the user is currently in.
    """
    typer.echo("Commit algorithm to API")


@algorithms_app.command("delete")
def delete_algorithm(remote: bool = typer.Option(False)) -> None:
    """Delete algorithm.

    The algorithm is deleted from the local disk. The algorithm is
    selected on the folder, the user is currently in. Based on the
    argument, the remote algorithm can also be deleted.
    """
    if remote:
        typer.echo("Delete local and remote algorithm")
    else:
        typer.echo("Delete local algorithm only")


@algorithms_app.command("describe")
def describe_algorithm(remote: bool = typer.Option(False)) -> None:
    """Describe algorithm.

    Describe the algorithm. Both metadata and data from the algorithm
    itself are shown. The algorithm is selected on the folder, the user
    is currently in. Based on the argument, the remote algorithm can
    also be described.
    """
    if remote:
        typer.echo("Describe remote algorithm")
    else:
        typer.echo("Describe local algorithm")


@algorithms_app.command("execute")
def execute_algorithm() -> None:
    """Execute algorithm.

    Send all selected algorithms to the configured execution environment
    (e.g. remote or native). If no algorithms are selected, the
    algorithm in the directory the user is currently in will be
    "selected" and executed.
    """
    typer.echo("Execute algorithm")


@algorithms_app.command("list")
def list_algorithms(
    local: bool = typer.Option(False), project: Optional[str] = typer.Option(None), remote: bool = typer.Option(False)
) -> None:
    """List algorithms.

    List all algorithms known on that environment for the currently
    selected project. If no environment is chosen (either local or
    remote), an empty list is shown. If both environments are chosen,
    both lists are merged with all algorithms being distinct. With the
    project flag, a different project can be selected to list all
    algorithms for.
    """
    if local and remote:
        target = "remote and local"
    elif remote:
        target = "remote"
    else:
        target = "local"

    if project is None:
        project = "default"
    typer.echo(f"List {target} algorithms for project '{project}'")


@algorithms_app.command("select")
def select_algorithm() -> None:
    """Select algorithm."""
    typer.echo("Select algorithm")


@configuration_app.command("get")
def get_config(key: str) -> None:
    """Get config."""
    typer.echo(f"Get config for '{key}'")


@configuration_app.command("list")
def list_config() -> None:
    """List config."""
    typer.echo("List config")


@configuration_app.command("set")
def set_config(key: str, value: str) -> None:
    """Set config."""
    typer.echo(f"Set config '{key}={value}'")


@projects_app.command("create")
def create_project(name: str) -> None:
    """Create project."""
    typer.echo(f"Create project '{name}'")


@projects_app.command("delete")
def delete_project(remote: bool = typer.Option(False)) -> None:
    """Delete project."""
    if remote:
        typer.echo("Delete remote and local project")
    else:
        typer.echo("Delete local project")


@projects_app.command("describe")
def describe_project(remote: bool = typer.Option(False)) -> None:
    """Describe project."""
    if remote:
        typer.echo("Describe remote project")
    else:
        typer.echo("Describe local project")


@projects_app.command("list")
def list_projects(local: bool = typer.Option(False), remote: bool = typer.Option(False)) -> None:
    """List project."""
    if local and remote:
        target = "remote and local"
    elif remote:
        target = "remote"
    else:
        target = "local"

    typer.echo(f"List {target} projects")


@projects_app.command("sync")
def sync_projects(dest: Destination = Destination.LOCAL) -> None:
    """Sync project."""
    typer.echo(f"Sync projects with {dest}")


@app.command("login")
def login(host: str) -> None:
    """Log in to Quantum Inspire."""
    typer.echo(f"Login to {host}")


@app.command("logout")
def logout(host: str) -> None:
    """Log out of Quantum Inspire."""
    typer.echo(f"Logout from {host}")
