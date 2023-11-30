class MMFileHandler:
    @staticmethod
    def readMMMetaData(path):
        fh = external.tifffile.tifffile.FileHandle(
            path, mode="rb", name=None, offset=None, size=None
        )
        md = external.tifffile.tifffile.read_micromanager_metadata(fh)
        fh.close()
        return md

    @staticmethod
    def mdToDF(micromanager_metadata_dict, p):
        md = micromanager_metadata_dict
        summary = md["summary"]
        channels = summary["ChNames"]
        try:
            ordered_positions = [
                (p["GridColumnIndex"], p["GridRowIndex"])
                for p in MMFileHandler.readMMMetaData(p)["summary"][
                    "InitialPositionList"
                ]
            ]
        except KeyError:
            # ordered_positions = [(p['GridRow'], p['GridCol'])
            ordered_positions = [
                (p["GridCol"], p["GridRow"])
                for p in MMFileHandler.readMMMetaData(p)["summary"]["StagePositions"]
            ]

        # Does the key "Inteval_ms" exist?
        if "Interval_ms" in summary.keys():
            fixedIntervalTimes = summary["Interval_ms"]
        else:
            fixedIntervalTimes = None

        frames = summary["Frames"]

        # handles case of single timepoint
        if len(summary["CustomIntervals_ms"]) == 0:
            delay_times = [0]
            customTimes_s = [0]
        else:
            delay_times = np.cumsum([t / 1000.0 for t in summary["CustomIntervals_ms"]])
            customTimes_s = [t / 1000.0 for t in summary["CustomIntervals_ms"]]

        dims = (summary["Width"], summary["Height"])
        channel_index = md["index_map"]["channel"]
        position = md["index_map"]["position"][0]

        baseRecord = {
            "num_frames_total": frames,
            "dims": dims,
            "position": position,
            "x": ordered_positions[position][0],
            "y": ordered_positions[position][1],
        }

        records = []
        for f in range(frames):
            for c in range(len(channels)):
                sliceRecord = {
                    "stack_index": f * len(channels) + c,
                    "channel": channels[c],
                    "time_interval": customTimes_s[f],
                    "delay_time": delay_times[f],
                }
                records.append({**sliceRecord, **baseRecord})

        metadata_df = pd.DataFrame(records)
        return metadata_df

    @staticmethod
    def parseStackMD(path):
        md = MMFileHandler.readMMMetaData(path)
        md_df = MMFileHandler.mdToDF(md, path)
        md_df["path"] = str(path)
        return md_df

    @staticmethod
    def parseMMStackedFolder(root, channelExposureMap, remapChannels=None):
        metadata = pd.concat(
            [
                MMFileHandler.parseStackMD(str(folder))
                for folder in pathlib.Path(str(root)).iterdir()
                if "tif" in folder.suffix
            ]
        )
        metadata["exp"] = metadata.channel.apply(lambda c: channelExposureMap[c])
        if remapChannels:
            metadata["channel"] = metadata.channel.apply(lambda c: remapChannels[c])
        return metadata
