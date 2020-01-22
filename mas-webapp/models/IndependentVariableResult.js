'use strict';

module.exports = (sequelize, DataTypes) => {
  const IndependentVariableResult = sequelize.define('IndependentVariableResult', {
  	Name: DataTypes.STRING,
    Coefficient: DataTypes.FLOAT,
    Pval: DataTypes.FLOAT,
    Transformations: DataTypes.STRING,
    VIF: DataTypes.STRING
  }, {
    timestamps: false,
    freezeTableName: true,
    tableName: 'IndependentVariableResult'
  });

  IndependentVariableResult.associate = function(models) {
    models.IndependentVariableResult.belongsTo(models.ModelRunDetail, {
      onDelete: "CASCADE",
      foreignKey: "ModelId"
    });
    models.IndependentVariableResult.belongsTo(models.RunDetail, {
      onDelete: "CASCADE",
      foreignKey: "RunId"
    });
  };
  return IndependentVariableResult;
};
