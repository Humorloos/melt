package de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.statement;

/**
 * Roles a neighbor can take (subject or object)
 */
public enum NeighborRole {
    SUBJECT("s"),
    OBJECT("o");
    private String role;

    NeighborRole(String role) {
        this.role = role;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }
}
