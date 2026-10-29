"""Tests for the SMD skeletal model format."""
import io

import pytest

from srctools import Angle, Vec
from srctools.smd import Bone, BoneFrame, Mesh, Triangle, Vertex


@pytest.mark.parametrize('weights', [
    (1.0,),
    (0.25, 0.75),
    (0.125, 0.25, 0.625),
], ids=['one-bone', 'two-bones', 'three-bones'])
def test_export_bone_weights(weights: tuple[float, ...]) -> None:
    """Exported vertices preserve their UVs and weights when read back in."""
    mesh = Mesh.blank('root')
    root = mesh.root_bone()
    bones = [root]
    for index in range(1, len(weights)):
        bone = Bone(f'joint_{index}', root)
        mesh.bones[bone.name] = bone
        mesh.animation[0].append(BoneFrame(bone, Vec(index, 0, 0), Angle()))
        bones.append(bone)

    links = list(zip(bones, weights))
    mesh.triangles.append(Triangle(
        'example/material',
        Vertex(Vec(0, 0, 0), Vec(0, 0, 1), 0.25, 0.5, links.copy()),
        Vertex(Vec(1, 0, 0), Vec(0, 0, 1), 0.75, -0.25, links.copy()),
        Vertex(Vec(0, 1, 0), Vec(0, 0, 1), 0.5, 1.0, links.copy()),
    ))

    buf = io.BytesIO()
    mesh.export(buf)
    buf.seek(0)
    result = Mesh.parse_smd(buf)

    assert result.bones.keys() == mesh.bones.keys()
    for bone in bones:
        assert result.bones[bone.name].parent == bone.parent
    assert len(result.triangles) == 1
    assert result.triangles[0].mat == 'example/material'
    for original, actual in zip(mesh.triangles[0], result.triangles[0]):
        assert actual.pos == original.pos
        assert actual.norm == original.norm
        assert actual.tex_u == pytest.approx(original.tex_u)
        assert actual.tex_v == pytest.approx(original.tex_v)
        assert [bone.name for bone, weight in actual.links] == [bone.name for bone in bones]
        assert [weight for bone, weight in actual.links] == pytest.approx(weights)
