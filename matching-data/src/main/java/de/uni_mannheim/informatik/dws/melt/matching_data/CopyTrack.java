package de.uni_mannheim.informatik.dws.melt.matching_data;

public class CopyTrack extends Track {
    public CopyTrack(String remoteLocation, String name, String version, boolean useDuplicateFreeStorageLayout, GoldStandardCompleteness goldStandardCompleteness, boolean skipTestsWithoutRefAlign) {
        super(remoteLocation, name, version, useDuplicateFreeStorageLayout, goldStandardCompleteness, skipTestsWithoutRefAlign);
    }

    @Override
    protected void downloadToCache() throws Exception {
    }
}
