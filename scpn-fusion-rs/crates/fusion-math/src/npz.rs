// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — NPZ archives
//! NumPy `.npz` archives: reading and writing named arrays.
//!
//! An `.npz` file is a ZIP archive whose members are `.npy` arrays, as written
//! by `numpy.savez` (stored members) and `numpy.savez_compressed` (deflated
//! members). Members are read in memory only; nothing is extracted to disk.
//!
//! The `.npy` encoding is provided by `ndarray-npy`; the archive layer uses the
//! maintained `zip` crate directly, so the archive handling follows its current
//! security fixes.

use ndarray::{ArrayBase, Data, DataOwned, Dimension};
use ndarray_npy::{
    ReadNpyError, ReadNpyExt, ReadableElement, WritableElement, WriteNpyError, WriteNpyExt,
};
use std::error::Error;
use std::fmt;
use std::io::{BufWriter, Read, Seek, Write};
use zip::result::ZipError;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// File-name suffix NumPy gives every array member of an archive.
const NPY_SUFFIX: &str = ".npy";

/// Failure while reading or writing an `.npz` archive.
#[derive(Debug)]
pub enum NpzError {
    /// The ZIP container is malformed, or the requested member is absent.
    Zip(ZipError),
    /// A member is not a valid `.npy` array of the requested type and rank.
    ReadNpy(ReadNpyError),
    /// An array could not be encoded as `.npy`.
    WriteNpy(WriteNpyError),
}

impl fmt::Display for NpzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zip(err) => write!(f, "npz archive error: {err}"),
            Self::ReadNpy(err) => write!(f, "npz member is not a readable npy array: {err}"),
            Self::WriteNpy(err) => write!(f, "array could not be written as npy: {err}"),
        }
    }
}

impl Error for NpzError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Zip(err) => Some(err),
            Self::ReadNpy(err) => Some(err),
            Self::WriteNpy(err) => Some(err),
        }
    }
}

impl From<ZipError> for NpzError {
    fn from(err: ZipError) -> Self {
        Self::Zip(err)
    }
}

impl From<ReadNpyError> for NpzError {
    fn from(err: ReadNpyError) -> Self {
        Self::ReadNpy(err)
    }
}

impl From<WriteNpyError> for NpzError {
    fn from(err: WriteNpyError) -> Self {
        Self::WriteNpy(err)
    }
}

/// Reader for the named arrays of an `.npz` archive.
pub struct NpzReader<R: Read + Seek> {
    archive: ZipArchive<R>,
}

impl<R: Read + Seek> NpzReader<R> {
    /// Open an archive from a seekable reader.
    ///
    /// # Errors
    /// Returns [`NpzError::Zip`] when the reader does not hold a ZIP archive.
    pub fn new(reader: R) -> Result<Self, NpzError> {
        Ok(Self {
            archive: ZipArchive::new(reader)?,
        })
    }

    /// Number of members in the archive.
    pub fn len(&self) -> usize {
        self.archive.len()
    }

    /// Whether the archive has no members.
    pub fn is_empty(&self) -> bool {
        self.archive.is_empty()
    }

    /// Array names in archive order, without the `.npy` suffix.
    ///
    /// # Errors
    /// Returns [`NpzError::Zip`] when a member header cannot be read.
    pub fn names(&mut self) -> Result<Vec<String>, NpzError> {
        let mut names = Vec::with_capacity(self.archive.len());
        for index in 0..self.archive.len() {
            let member = self.archive.by_index(index)?;
            let name = member.name()?;
            names.push(name.strip_suffix(NPY_SUFFIX).unwrap_or(&name).to_owned());
        }
        Ok(names)
    }

    /// Read the array stored under `name`.
    ///
    /// The member is looked up first under `name` exactly and then under
    /// `name` with the `.npy` suffix, which is how NumPy names the members.
    ///
    /// # Errors
    /// Returns [`NpzError::Zip`] when neither member exists or the archive is
    /// damaged, and [`NpzError::ReadNpy`] when the member's element type or
    /// rank differs from the requested array type.
    pub fn by_name<S, D>(&mut self, name: &str) -> Result<ArrayBase<S, D>, NpzError>
    where
        S: DataOwned,
        S::Elem: ReadableElement,
        D: Dimension,
    {
        match self.archive.by_name(name) {
            Ok(member) => return Ok(ArrayBase::<S, D>::read_npy(member)?),
            Err(ZipError::FileNotFound) => {}
            Err(err) => return Err(err.into()),
        }
        let member = self.archive.by_name(&format!("{name}{NPY_SUFFIX}"))?;
        Ok(ArrayBase::<S, D>::read_npy(member)?)
    }
}

/// Writer that adds named arrays to a new `.npz` archive.
pub struct NpzWriter<W: Write + Seek> {
    archive: ZipWriter<W>,
    options: SimpleFileOptions,
}

impl<W: Write + Seek> NpzWriter<W> {
    /// Start an archive with stored (uncompressed) members, as `numpy.savez`.
    pub fn new(writer: W) -> Self {
        Self::with_compression(writer, CompressionMethod::Stored)
    }

    /// Start an archive with deflated members, as `numpy.savez_compressed`.
    pub fn new_compressed(writer: W) -> Self {
        Self::with_compression(writer, CompressionMethod::Deflated)
    }

    fn with_compression(writer: W, method: CompressionMethod) -> Self {
        Self {
            archive: ZipWriter::new(writer),
            options: SimpleFileOptions::default().compression_method(method),
        }
    }

    /// Add `array` under `name`; the member is stored as `name.npy`.
    ///
    /// # Errors
    /// Returns [`NpzError::Zip`] when the member cannot be started (for
    /// example a duplicate name) and [`NpzError::WriteNpy`] when the array
    /// cannot be encoded.
    pub fn add_array<N, S, D>(&mut self, name: N, array: &ArrayBase<S, D>) -> Result<(), NpzError>
    where
        N: Into<String>,
        S: Data,
        S::Elem: WritableElement,
        D: Dimension,
    {
        self.archive
            .start_file(name.into() + NPY_SUFFIX, self.options)?;
        array.write_npy(BufWriter::new(&mut self.archive))?;
        Ok(())
    }

    /// Write the central directory and return the underlying writer.
    ///
    /// # Errors
    /// Returns [`NpzError::Zip`] when the archive cannot be completed.
    pub fn finish(self) -> Result<W, NpzError> {
        let mut writer = self.archive.finish()?;
        writer.flush().map_err(ZipError::from)?;
        Ok(writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2, Array3, ArrayD, Ix1, Ix2, IxDyn, OwnedRepr};
    use std::fs::File;
    use std::io::Cursor;
    use std::path::PathBuf;

    fn repo_file(relative: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join(relative)
    }

    fn round_trip(compressed: bool) -> NpzReader<Cursor<Vec<u8>>> {
        let buffer = Cursor::new(Vec::new());
        let mut writer = if compressed {
            NpzWriter::new_compressed(buffer)
        } else {
            NpzWriter::new(buffer)
        };
        writer
            .add_array("vector", &array![1.5_f64, -2.25, 3.0])
            .unwrap();
        writer
            .add_array("matrix", &array![[1_i32, 2, 3], [4, 5, 6]])
            .unwrap();
        writer
            .add_array("cube", &Array3::<f64>::from_elem((2, 3, 4), 0.125))
            .unwrap();
        let bytes = writer.finish().unwrap().into_inner();
        NpzReader::new(Cursor::new(bytes)).unwrap()
    }

    #[test]
    fn stored_and_deflated_archives_round_trip_every_array() {
        for compressed in [false, true] {
            let mut reader = round_trip(compressed);
            assert_eq!(reader.len(), 3);
            assert!(!reader.is_empty());
            assert_eq!(reader.names().unwrap(), ["vector", "matrix", "cube"]);
            let vector: Array1<f64> = reader.by_name("vector").unwrap();
            assert_eq!(vector, array![1.5, -2.25, 3.0]);
            let matrix: Array2<i32> = reader.by_name("matrix.npy").unwrap();
            assert_eq!(matrix, array![[1, 2, 3], [4, 5, 6]]);
            let cube: Array3<f64> = reader.by_name("cube").unwrap();
            assert_eq!(cube.dim(), (2, 3, 4));
            assert!(cube.iter().all(|value| *value == 0.125));
        }
    }

    #[test]
    fn numpy_savez_archive_is_read_exactly() {
        // Written by numpy.savez: stored members.
        let path = repo_file("weights/pretrained_mlp_itpa.npz");
        let mut reader = NpzReader::new(File::open(path).unwrap()).unwrap();
        assert_eq!(reader.len(), 8);
        let mean: Array1<f64> = reader.by_name("feature_mean").unwrap();
        assert_eq!(mean.len(), 9);
        assert_eq!(mean[0], 1.8371698113207546);
        assert_eq!(mean[1], 2.926792452830189);
        let w1: Array2<f64> = reader.by_name("w1").unwrap();
        assert_eq!(w1.dim(), (9, 32));
        assert_eq!(w1[[8, 31]], 0.06634665179038143);
        let b2: ArrayD<f64> = reader.by_name("b2").unwrap();
        assert_eq!(b2.ndim(), 0);
        assert_eq!(b2[IxDyn(&[])], 0.17293195782675738);
    }

    #[test]
    fn numpy_savez_compressed_archive_is_read_exactly() {
        // Written by numpy.savez_compressed: deflated members.
        let path =
            repo_file("validation/reference_data/full_fidelity_public_artifacts/dream_avalanche_public_raw.npz");
        let mut reader = NpzReader::new(File::open(path).unwrap()).unwrap();
        assert_eq!(
            reader.names().unwrap(),
            ["EOverEcTot", "GammaNum", "nD", "nZ"]
        );
        let n_d: Array1<f64> = reader.by_name("nD").unwrap();
        assert_eq!(n_d, array![1e19, 1e21]);
        let e_over_ec: Array1<f64> = reader.by_name("EOverEcTot").unwrap();
        assert_eq!(e_over_ec, array![5.0, 30.0, 100.0]);
        let gamma: ArrayD<f64> = reader.by_name("GammaNum").unwrap();
        assert_eq!(gamma.shape(), [3, 2, 2, 2, 2]);
        assert_eq!(gamma[IxDyn(&[2, 1, 1, 1, 1])], 7440.083881718525);
    }

    #[test]
    fn missing_member_is_a_zip_not_found_error() {
        let mut reader = round_trip(false);
        let err = reader.by_name::<OwnedRepr<f64>, Ix1>("absent").unwrap_err();
        assert!(matches!(err, NpzError::Zip(ZipError::FileNotFound)));
        assert!(err.to_string().starts_with("npz archive error"));
        assert!(err.source().is_some());
    }

    #[test]
    fn wrong_rank_or_type_is_a_read_npy_error() {
        let mut reader = round_trip(false);
        let rank = reader.by_name::<OwnedRepr<f64>, Ix2>("vector").unwrap_err();
        assert!(matches!(rank, NpzError::ReadNpy(_)));
        let element = reader.by_name::<OwnedRepr<f64>, Ix2>("matrix").unwrap_err();
        assert!(matches!(element, NpzError::ReadNpy(_)));
        assert!(element.to_string().contains("not a readable npy array"));
        assert!(element.source().is_some());
    }

    #[test]
    fn non_archive_input_is_refused() {
        let err = NpzReader::new(Cursor::new(b"not a zip archive".to_vec()))
            .err()
            .unwrap();
        assert!(matches!(err, NpzError::Zip(_)));
    }

    #[test]
    fn duplicate_member_name_is_refused() {
        let mut writer = NpzWriter::new(Cursor::new(Vec::new()));
        writer.add_array("x", &array![1.0_f64]).unwrap();
        let err = writer.add_array("x", &array![2.0_f64]).unwrap_err();
        assert!(matches!(err, NpzError::Zip(_)));
    }

    #[test]
    fn empty_archive_has_no_names() {
        let bytes = NpzWriter::new(Cursor::new(Vec::new()))
            .finish()
            .unwrap()
            .into_inner();
        let mut reader = NpzReader::new(Cursor::new(bytes)).unwrap();
        assert!(reader.is_empty());
        assert!(reader.names().unwrap().is_empty());
    }

    #[test]
    fn write_npy_error_displays_its_cause() {
        let err = NpzError::from(WriteNpyError::Io(std::io::Error::other("disk full")));
        assert!(err
            .to_string()
            .starts_with("array could not be written as npy"));
        assert!(err.source().is_some());
    }
}
