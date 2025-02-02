package de.uni_mannheim.informatik.dws.melt.matching_data;

/**
 * Defines how complete a gold standard is.
 * It could be complete (all correspondences *NOT* in gold standard are wrong).
 * It could be partial (cannot say something about correspondences not in GS).
 * For a partial GS there are more sub cases.
 * Assume &lt;a,b,=,1.0&gt; in the GS.
 * <ul>
 * <li>Then system correspondence &lt;a,c,=,1.0&gt; is assumed to be wrong if source is complete.</li>
 * <li>Then system correspondence &lt;c,b,=,1.0&gt; is assumed to be wrong if target is complete.</li>
 * </ul>
 * If both are incomplete nothing can be said about &lt;a,c,=,1.0&gt; or &lt;c,b,=,1.0&gt;.
 */
public enum GoldStandardCompleteness {


    /**
     * The GS is complete and thus all correspondences <em>NOT</em> in gold standard are wrong.
     */
    COMPLETE,

    /**
     * The gold standard is not complete, but the matched elements are.
     * For example if there is a correspondence {@code <a, b, =>}, each match involving {@code a} that is not
     * in the reference is incorrect and the same accounts for {@code b}.
     */
    PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE,

    /**
     * Every correspondence involving an element of the complete side that is not available in the reference can be regarded as wrong.
     */
    PARTIAL_SOURCE_COMPLETE_TARGET_INCOMPLETE,

    /**
     * Every correspondence involving an element of the complete side that is not available in the reference can be regarded as wrong.
     */
    PARTIAL_SOURCE_INCOMPLETE_TARGET_COMPLETE,

    /**
     * Partial GS where correspondences NOT in GS cannot be judged.
     */
    PARTIAL_SOURCE_INCOMPLETE_TARGET_INCOMPLETE;
    
    /**
     * Returns true if the GS is complete and thus all correspondences *NOT* in gold standard are wrong.
     * @return true if the GS is complete and thus all correspondences *NOT* in gold standard are wrong
     */
    public boolean isGoldStandardComplete(){
        if(this == GoldStandardCompleteness.COMPLETE)
            return true;
        return false;
    }
    
    /**
     * If the source is complete, there are no further correct correspondences
     * concerning one source element except of that/those in the gold standard.
     * 
     * Example:
     * <ul>
     *   <li>Alignment: &lt;a,b,=,1.0&gt;</li>
     *   <li>System: &lt;a,b,=,1.0&gt;, &lt;a,c,=,1.0&gt;</li>
     * </ul>
     * The systems result is interpreted as follows:
     * TP: &lt;a,b,=,1.0&gt;
     * FP: &lt;a,c,=,1.0&gt;
     * Every other/additional involving 'a' would be regarded as FP.
     * @return true, if there are no further correct correspondences
     * concerning one source element except of that/those in the gold standard
     */
    public boolean isSourceComplete(){
        if(this==PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE ||
           this == PARTIAL_SOURCE_COMPLETE_TARGET_INCOMPLETE ||
           this == COMPLETE)
            return true;
        return false;
    }
    
    /**
     * If the target is complete, there are no further correct correspondences
     * concerning one target element except of that/those in the gold standard.
     * 
     * Example:
     * <ul>
     *  <li>Alignment: &lt;a,b,=,1.0&gt;</li>
     *  <li>System: &lt;a,b,=,1.0&gt;, &lt;c, b,=,1.0&gt;</li>
     * </ul>
     * The systems result is interpreted as follows:
     * TP: &lt;a,b,=,1.0&gt;
     * FP: &lt;c,b,=,1.0&gt;
     * Every other/additional involving 'b' would be regarded as FP.
     * @return true, if there are no further correct correspondences
     * concerning one target element exept of that/those in the gold standard.
     */
    public boolean isTargetComplete(){
        if(this == PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE ||
           this == PARTIAL_SOURCE_INCOMPLETE_TARGET_COMPLETE ||
           this == COMPLETE)
            return true;
        return false;
    }
}
