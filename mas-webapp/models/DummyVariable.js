'use strict';

module.exports = (sequelize, DataTypes) => {
  const DummyVariable = sequelize.define('DummyVariable', {
  	Name: DataTypes.STRING,
    Coefficient: DataTypes.FLOAT,
    Pval: DataTypes.FLOAT
  }, {
    timestamps: false,
    freezeTableName: true,
    tableName: 'DummyVariable'
  });

  DummyVariable.associate = function(models) {
    models.DummyVariable.belongsTo(models.RunDetail, {
      onDelete: "CASCADE",
      foreignKey: "RunId"
    });
  };
  return DummyVariable;
};