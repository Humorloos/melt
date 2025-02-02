package de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.statement;

import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.processedNode.ProcessedResource;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.rdf.model.Statement;

/**
 * representation of a subject statement
 */
public class SubjectStatement extends ProcessedStatement<ProcessedResource<Resource>> {

    public SubjectStatement(Statement statement) {
        super(statement);
        this.neighbor = new ProcessedResource<>(statement.getSubject());
        this.role = NeighborRole.SUBJECT;
    }

}
