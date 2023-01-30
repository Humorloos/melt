package de.uni_mannheim.informatik.dws.melt.matching_data;

/**
 * Class for copying matching tracks
 */
public class CopyTrack extends LocalTrack {
    public CopyTrack(String name, String version, GoldStandardCompleteness goldStandardCompleteness) {
        super(name, version, goldStandardCompleteness);
    }

    @Override
    protected void downloadToCache() throws Exception {
    }
}
