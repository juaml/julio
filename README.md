# junifer-julio

Junifer feature registry: publish the features computed with
[junifer](https://github.com/juaml/junifer) so that others can find and use
them.

A registry is a [DataLad](https://www.datalad.org) dataset with one file per
feature, the junifer YAML to compute it again and its metadata:

```
registry/
├── features/
│   ├── feature-<md5>.hdf5        # the feature, readable with junifer
│   ├── feature-<md5>.yaml        # the YAML to compute it (generate_yaml)
│   ├── feature-<md5>-meta.yaml   # the metadata, used by the site
│   └── ...
└── registry-config.yaml          # the mappings to human-readable names
```

## Installation

```
pip install julio
```

julio needs [git-annex](https://git-annex.branchable.com) and features
computed with junifer >= 0.0.9. DataLad's scripts (e.g., `datalad` and
`git-annex-remote-ora`, to use RIA stores) must be in `PATH`: install julio
in an environment that is activated, not with tools that only expose its
command (e.g., `pipx`).

## Usage

The commands are available as `julio <command>` and as
`junifer registry <command>`.

Create a registry:

```
julio create my-registry
```

Add the features computed with a junifer YAML (all the features in its
storage, once collected); the registry can be a path or the URL of a remote
registry, which is cloned, updated and pushed:

```
julio add features.yaml --registry my-registry
```

Publish the registry to a sibling, a RIA store (with the data) or a git URL:

```
julio publish --registry my-registry --remote ria+ssh://server/path/to/store
```

Build the static site to find and download the features:

```
julio build --registry my-registry --output site
```

The download instructions of the site clone the registry from its `origin`
sibling (e.g., the one added by `publish`).

## Mappings

The site shows human-readable names, descriptions and tags for the datasets
and markers, from the `mappings` in `registry-config.yaml`. `create` writes
the default mappings of julio, and they are part of the registry: updating
julio does not change them, and they can be edited, e.g., to add custom
datagrabbers:

```yaml
mappings:
  datagrabbers:
    - class: MyDataGrabber
      display_name: My Dataset
      tags:
        - My Dataset
      description: The dataset of my lab
  markers:
    - ...
```

The mappings are applied when building the site, so the changes apply to the
features already in the registry. The components not in the mappings are
shown with their class name.
